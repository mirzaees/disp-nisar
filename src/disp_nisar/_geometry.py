"""Module for computing geometry products: incidence angles, LOS vectors, and layover/shadow masks."""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import rioxarray as rxr
from dolphin._types import Filename
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _interp_chunk(args):
    """Interpolate a chunk of DEM onto 3D radar grid to extract surface values.

    Parameters
    ----------
    args : tuple
        (row_slice, y_vals, x_vals, dem_chunk, heights, y_rg_flip, x_rg,
         data_3d_list, src_epsg, dem_crs_str)

    Returns
    -------
    tuple
        (row_slice, (incidence_chunk, los_east_chunk, los_north_chunk))
    """
    (
        row_slice,
        y_vals,
        x_vals,
        dem_chunk,
        heights,
        y_rg_flip,
        x_rg,
        data_3d_list,
        src_epsg,
        dem_crs_str,
    ) = args

    # Transform DEM coordinates to radar grid CRS
    t_to_rg = Transformer.from_crs(dem_crs_str, f"EPSG:{src_epsg}", always_xy=True)
    xx, yy = np.meshgrid(x_vals, y_vals, indexing="xy")
    x_rg_coords, y_rg_coords = t_to_rg.transform(xx.ravel(), yy.ravel())
    x_rg_coords = x_rg_coords.reshape(xx.shape)
    y_rg_coords = y_rg_coords.reshape(yy.shape)

    # Prepare interpolators for each 3D dataset
    interpolators = []
    for data_3d in data_3d_list:
        interp = RegularGridInterpolator(
            (heights, y_rg_flip, x_rg),
            data_3d,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        interpolators.append(interp)

    # Create query points: (height, y, x)
    dem_flat = dem_chunk.ravel()
    y_flat = y_rg_coords.ravel()
    x_flat = x_rg_coords.ravel()
    query_points = np.column_stack([dem_flat, y_flat, x_flat])

    # Interpolate each dataset
    results = []
    for interp in interpolators:
        vals = interp(query_points)
        vals_reshaped = vals.reshape(dem_chunk.shape).astype(np.float32)
        results.append(vals_reshaped)

    return (row_slice, tuple(results))


def prepare_geometry_layers(
    gslc_path: Filename,
    dem_path: Filename,
    output_dir: Path,
    template_raster: Filename,
    incidence_output_name: str = "incidence_angle.tif",
    los_east_output_name: str = "los_east.tif",
    los_north_output_name: str = "los_north.tif",
    layover_shadow_output_name: str = "layover_shadow_mask.tif",
    chunk_size: int = 200,
    n_workers: int = 8,
) -> dict[str, Path]:
    """Prepare geometry layers from GSLC radar grid and DEM at full frame resolution.

    Creates geometry layers at the same exact grid as the GSLC frame (matching
    the nodata mask and other frame-level products). These full-resolution layers are:
    - Used for masking during phase linking (layover/shadow)
    - Downsampled later for product generation (incidence angles, LOS)

    Computes:
    - Incidence angle at surface (full frame resolution)
    - LOS unit vectors east/north components (full frame resolution)
    - Layover/shadow mask (full frame resolution, for use in block processing)

    Parameters
    ----------
    gslc_path : Filename
        Path to NISAR GSLC HDF5 file containing radar grid metadata
    dem_path : Filename
        Path to DEM file (GeoTIFF)
    output_dir : Path
        Directory to save output files
    template_raster : Filename
        Template raster defining the target frame grid (e.g., a GSLC data layer)
        Geometry layers will be created at this exact grid.
    incidence_output_name : str
        Output filename for incidence angle raster
    los_east_output_name : str
        Output filename for LOS east component
    los_north_output_name : str
        Output filename for LOS north component
    layover_shadow_output_name : str
        Output filename for layover/shadow mask
    chunk_size : int
        Number of rows to process at once
    n_workers : int
        Number of parallel workers

    Returns
    -------
    dict[str, Path]
        Dictionary with keys:
        - 'incidence_angle': Path to incidence angle file (full frame resolution)
        - 'los_east': Path to LOS east file (full frame resolution)
        - 'los_north': Path to LOS north file (full frame resolution)
        - 'layover_shadow_mask': Path to layover/shadow mask file (full frame resolution)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    incidence_path = output_dir / incidence_output_name
    los_east_path = output_dir / los_east_output_name
    los_north_path = output_dir / los_north_output_name
    layover_shadow_path = output_dir / layover_shadow_output_name

    # Check if outputs already exist
    if all(
        p.exists()
        for p in [incidence_path, los_east_path, los_north_path, layover_shadow_path]
    ):
        logger.info("All geometry layers already exist, skipping computation")
        return {
            "incidence_angle": incidence_path,
            "los_east": los_east_path,
            "los_north": los_north_path,
            "layover_shadow_mask": layover_shadow_path,
        }

    logger.info(f"Reading radar grid from {gslc_path}")
    # Read radar grid data from GSLC
    with h5py.File(gslc_path) as f:
        rg = f["science/LSAR/GSLC/metadata/radarGrid"]
        heights = rg["heightAboveEllipsoid"][:]
        x_rg = rg["xCoordinates"][:]
        y_rg = rg["yCoordinates"][:]
        inc_3d = rg["incidenceAngle"][:]
        los_x_3d = rg["losUnitVectorX"][:]
        los_y_3d = rg["losUnitVectorY"][:]
        src_epsg = int(rg["projection"].attrs["epsg_code"])

    logger.info(f"Radar grid CRS: EPSG:{src_epsg}")
    logger.info(f"Radar grid x: {x_rg.min():.0f} – {x_rg.max():.0f}")
    logger.info(f"Radar grid y: {y_rg.min():.0f} – {y_rg.max():.0f}")

    # Validate inputs
    if not (np.all(np.diff(heights) > 0)):
        raise ValueError("Heights must be strictly increasing")
    if not (np.all(np.diff(x_rg) > 0)):
        raise ValueError("x_rg must be strictly increasing")

    # Check coordinate overlap
    t_wgs84 = Transformer.from_crs(f"EPSG:{src_epsg}", "EPSG:4326", always_xy=True)
    lons, lats = t_wgs84.transform(
        [x_rg.min(), x_rg.max()], [y_rg.min(), y_rg.max()]
    )
    logger.info(
        f"Radar grid WGS84: lon {min(lons):.2f}–{max(lons):.2f}, "
        f"lat {min(lats):.2f}–{max(lats):.2f}"
    )

    # Load template raster to get target frame grid
    logger.info(f"Loading template raster to define output grid: {template_raster}")
    template_da = rxr.open_rasterio(template_raster, masked=True).squeeze()
    target_crs = template_da.rio.crs
    target_shape = template_da.shape
    logger.info(
        f"Target frame grid: {target_shape[0]} x {target_shape[1]} pixels, "
        f"CRS: {target_crs}"
    )

    # Reproject DEM using fast GDAL warp (cached on disk)
    logger.info(f"Reprojecting DEM from {dem_path}")
    reprojected_dem_path = output_dir / "dem_reprojected.tif"

    if reprojected_dem_path.exists():
        logger.info(f"Using cached reprojected DEM: {reprojected_dem_path}")
    else:
        from osgeo import gdal

        # Get template bounds and resolution
        bounds = template_da.rio.bounds()
        gt = template_da.rio.transform()
        x_res = gt.a
        y_res = -gt.e  # Negative because gt.e is negative for north-up

        logger.info(
            f"Warping DEM to match frame: {target_shape[1]}x{target_shape[0]}, "
            f"resolution: {x_res:.2f}m"
        )

        # Use GDAL warp (much faster than rioxarray for large DEMs)
        warp_options = gdal.WarpOptions(
            format="GTiff",
            dstSRS=str(target_crs),
            outputBounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
            xRes=x_res,
            yRes=y_res,
            resampleAlg="bilinear",
            creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
            multithread=True,
            warpMemoryLimit=512,  # MB
        )

        gdal.Warp(str(reprojected_dem_path), str(dem_path), options=warp_options)
        logger.info(f"Saved reprojected DEM to {reprojected_dem_path}")

    # Load the reprojected DEM
    dem_da = rxr.open_rasterio(reprojected_dem_path, masked=True).squeeze()
    dem_val = dem_da.values
    dem_crs = target_crs

    # Clean up
    del template_da

    # Flip y-axis for interpolation (radar grid may be top-to-bottom)
    y_rg_flip = y_rg[::-1]
    inc_3d_flip = inc_3d[:, ::-1, :]
    los_x_3d_flip = los_x_3d[:, ::-1, :]
    los_y_3d_flip = los_y_3d[:, ::-1, :]

    # Prepare chunks for parallel processing
    dem_crs_str = str(dem_crs)
    n_rows = dem_da.shape[0]
    slices = [slice(i, min(i + chunk_size, n_rows)) for i in range(0, n_rows, chunk_size)]
    tasks = [
        (
            sl,
            dem_da.y.values[sl],
            dem_da.x.values,
            dem_val[sl],
            heights,
            y_rg_flip,
            x_rg,
            [inc_3d_flip, los_x_3d_flip, los_y_3d_flip],
            src_epsg,
            dem_crs_str,
        )
        for sl in slices
    ]

    # Initialize output arrays
    inc_surface = np.full(dem_da.shape, np.nan, dtype="float32")
    los_east_surf = np.full(dem_da.shape, np.nan, dtype="float32")
    los_north_surf = np.full(dem_da.shape, np.nan, dtype="float32")

    # Process in parallel with optimized chunking
    logger.info(f"Interpolating geometry data with {n_workers} workers")
    chunksize = max(1, len(tasks) // (n_workers * 4))  # Optimize work distribution
    with mp.get_context("fork").Pool(processes=n_workers) as pool:
        for sl, results in tqdm(
            pool.imap(_interp_chunk, tasks, chunksize=chunksize),
            total=len(tasks),
            desc="Geometry interpolation",
        ):
            inc_surface[sl] = results[0]
            los_east_surf[sl] = results[1]
            los_north_surf[sl] = results[2]

    # Compute LOS up component (to complete unit vector)
    los_up_surf = np.sqrt(
        np.clip(1.0 - los_east_surf**2 - los_north_surf**2, 0, None)
    ).astype("float32")

    # Save all geometry layers at DEM (full) resolution using fast GDAL writes
    logger.info("Saving geometry layers at full resolution (DEM grid)")

    from osgeo import gdal, osr

    # Get geotransform and projection from DEM
    gt = dem_da.rio.transform().to_gdal()
    proj = osr.SpatialReference()
    proj.ImportFromWkt(dem_crs.to_wkt())

    # Helper function for fast GDAL writes
    def _write_geotiff_fast(array, output_path, dtype, nodata=None):
        driver = gdal.GetDriverByName("GTiff")
        if dtype == gdal.GDT_Float32:
            gdal_dtype = gdal.GDT_Float32
        elif dtype == gdal.GDT_Byte:
            gdal_dtype = gdal.GDT_Byte
        else:
            gdal_dtype = gdal.GDT_Float32

        ds = driver.Create(
            str(output_path),
            array.shape[1],
            array.shape[0],
            1,
            gdal_dtype,
            options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
        )
        ds.SetGeoTransform(gt)
        ds.SetProjection(proj.ExportToWkt())
        band = ds.GetRasterBand(1)
        if nodata is not None:
            band.SetNoDataValue(nodata)
        band.WriteArray(array)
        band.FlushCache()
        ds = None

    # Save incidence angle
    logger.info(f"Saving incidence angle to {incidence_path}")
    _write_geotiff_fast(inc_surface, incidence_path, gdal.GDT_Float32, nodata=np.nan)
    logger.info(
        f"  Incidence angle range: {np.nanmin(inc_surface):.2f}–{np.nanmax(inc_surface):.2f} deg"
    )

    # Save LOS east component
    logger.info(f"Saving LOS east to {los_east_path}")
    _write_geotiff_fast(los_east_surf, los_east_path, gdal.GDT_Float32, nodata=np.nan)
    logger.info(
        f"  LOS east range: {np.nanmin(los_east_surf):.3f}–{np.nanmax(los_east_surf):.3f}"
    )

    # Save LOS north component
    logger.info(f"Saving LOS north to {los_north_path}")
    _write_geotiff_fast(los_north_surf, los_north_path, gdal.GDT_Float32, nodata=np.nan)
    logger.info(
        f"  LOS north range: {np.nanmin(los_north_surf):.3f}–{np.nanmax(los_north_surf):.3f}"
    )

    # Compute and save layover/shadow mask
    logger.info("Computing layover/shadow mask")
    layover_shadow_mask = _compute_layover_shadow_mask(
        inc_surface, dem_matched=dem_val
    )

    # Save as uint8: 0=bad (layover/shadow), 1=good
    _write_geotiff_fast(
        layover_shadow_mask.astype(np.uint8),
        layover_shadow_path,
        gdal.GDT_Byte,
        nodata=255,
    )
    logger.info(f"Saved layover/shadow mask to {layover_shadow_path}")
    pct_bad = 100 * (1 - layover_shadow_mask.mean())
    logger.info(f"  {pct_bad:.1f}% pixels masked as layover/shadow")

    return {
        "incidence_angle": incidence_path,
        "los_east": los_east_path,
        "los_north": los_north_path,
        "layover_shadow_mask": layover_shadow_path,
    }


def downsample_geometry_for_products(
    incidence_angle_path: Path,
    los_east_path: Path,
    los_north_path: Path,
    reference_raster: Filename,
    output_dir: Path,
) -> dict[str, Path]:
    """Downsample full-resolution geometry layers to match product grid.

    Parameters
    ----------
    incidence_angle_path : Path
        Path to full-resolution incidence angle raster
    los_east_path : Path
        Path to full-resolution LOS east raster
    los_north_path : Path
        Path to full-resolution LOS north raster
    reference_raster : Filename
        Reference raster defining the target grid (e.g., unwrapped phase product)
    output_dir : Path
        Directory to save downsampled geometry layers

    Returns
    -------
    dict[str, Path]
        Dictionary with keys:
        - 'incidence_angle': Path to downsampled incidence angle
        - 'los_east': Path to downsampled LOS east
        - 'los_north': Path to downsampled LOS north
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference raster to get target grid
    logger.info(f"Downsampling geometry layers to match {reference_raster}")

    from osgeo import gdal

    # Helper function for fast GDAL downsampling with caching
    def _downsample_with_gdal(input_path, output_path, reference_raster):
        """Use GDAL warp for fast downsampling."""
        if output_path.exists():
            logger.info(f"Using cached downsampled file: {output_path}")
            return output_path

        # Get reference grid info
        ref_ds = gdal.Open(str(reference_raster))
        ref_gt = ref_ds.GetGeoTransform()
        ref_proj = ref_ds.GetProjection()
        ref_xsize = ref_ds.RasterXSize
        ref_ysize = ref_ds.RasterYSize
        ref_ds = None

        # Calculate bounds
        left = ref_gt[0]
        top = ref_gt[3]
        right = left + ref_xsize * ref_gt[1]
        bottom = top + ref_ysize * ref_gt[5]

        # Use GDAL warp for fast resampling
        warp_options = gdal.WarpOptions(
            format="GTiff",
            dstSRS=ref_proj,
            outputBounds=(left, bottom, right, top),
            width=ref_xsize,
            height=ref_ysize,
            resampleAlg="bilinear",
            creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
            multithread=True,
            warpMemoryLimit=512,
        )

        gdal.Warp(str(output_path), str(input_path), options=warp_options)
        logger.info(f"Downsampled to {output_path}")
        return output_path

    outputs = {}

    # Downsample incidence angle
    inc_out = output_dir / "incidence_angle_downsampled.tif"
    _downsample_with_gdal(incidence_angle_path, inc_out, reference_raster)
    outputs["incidence_angle"] = inc_out

    # Downsample LOS east
    los_e_out = output_dir / "los_east_downsampled.tif"
    _downsample_with_gdal(los_east_path, los_e_out, reference_raster)
    outputs["los_east"] = los_e_out

    # Downsample LOS north
    los_n_out = output_dir / "los_north_downsampled.tif"
    _downsample_with_gdal(los_north_path, los_n_out, reference_raster)
    outputs["los_north"] = los_n_out

    return outputs


def _compute_layover_shadow_mask(
    incidence_angle: np.ndarray,
    dem_matched: np.ndarray,
    shadow_threshold_deg: float = 85.0,
    layover_slope_threshold: float = 0.5,
) -> np.ndarray:
    """Compute layover and shadow mask from incidence angle and DEM.

    Parameters
    ----------
    incidence_angle : np.ndarray
        Incidence angle in degrees
    dem_matched : np.ndarray
        DEM elevations (same grid as incidence angle)
    shadow_threshold_deg : float
        Incidence angles above this threshold are considered shadow
    layover_slope_threshold : float
        Local slope threshold for detecting layover (in DEM gradient units)

    Returns
    -------
    np.ndarray
        Binary mask: 1=good pixel, 0=bad (layover or shadow)
    """
    # Shadow: very steep incidence angles (near-horizontal look)
    is_shadow = incidence_angle > shadow_threshold_deg

    # Layover: steep terrain slopes toward radar (DEM gradient analysis)
    # Compute DEM gradient magnitude
    try:
        from scipy.ndimage import sobel

        dy = sobel(dem_matched, axis=0, mode="constant", cval=0.0)
        dx = sobel(dem_matched, axis=1, mode="constant", cval=0.0)
        slope_magnitude = np.sqrt(dx**2 + dy**2)
        is_layover = slope_magnitude > layover_slope_threshold
    except Exception as e:
        logger.warning(f"Failed to compute layover from DEM gradient: {e}")
        is_layover = np.zeros_like(is_shadow, dtype=bool)

    # Combine: any pixel with layover or shadow is masked (0)
    bad_pixels = is_shadow | is_layover

    # Good pixels = 1, bad = 0
    mask = (~bad_pixels).astype(np.uint8)

    # Handle NaN in incidence angle
    mask[np.isnan(incidence_angle)] = 0

    return mask
