"""Module for computing geometry products: incidence angles, LOS vectors, and layover/shadow masks."""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import rioxarray as rxr
import xarray as xr
from dolphin._types import Filename
from pyproj import Transformer
from rasterio.enums import Resampling
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
    incidence_output_name: str = "incidence_angle.tif",
    los_east_output_name: str = "los_east.tif",
    los_north_output_name: str = "los_north.tif",
    layover_shadow_output_name: str = "layover_shadow_mask.tif",
    chunk_size: int = 200,
    n_workers: int = 8,
) -> dict[str, Path]:
    """Prepare geometry layers from GSLC radar grid and DEM at full resolution.

    Creates geometry layers at the same resolution and grid as the input DEM,
    which should match the GSLC resolution. These full-resolution layers are:
    - Used for masking during phase linking (layover/shadow)
    - Downsampled later for product generation (incidence angles, LOS)

    Computes:
    - Incidence angle at surface (full resolution)
    - LOS unit vectors east/north components (full resolution)
    - Layover/shadow mask (full resolution, for use in block processing)

    Parameters
    ----------
    gslc_path : Filename
        Path to NISAR GSLC HDF5 file containing radar grid metadata
    dem_path : Filename
        Path to DEM file (GeoTIFF) at GSLC resolution
    output_dir : Path
        Directory to save output files
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
        - 'incidence_angle': Path to incidence angle file (full resolution)
        - 'los_east': Path to LOS east file (full resolution)
        - 'los_north': Path to LOS north file (full resolution)
        - 'layover_shadow_mask': Path to layover/shadow mask file (full resolution)
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

    # Load DEM
    logger.info(f"Loading DEM from {dem_path}")
    dem_da = rxr.open_rasterio(dem_path, masked=True).squeeze()
    dem_val = dem_da.values
    dem_crs = dem_da.rio.crs

    # Check DEM overlap with radar grid
    t_to_rg = Transformer.from_crs(dem_crs, f"EPSG:{src_epsg}", always_xy=True)
    dem_xs = [float(dem_da.x.min()), float(dem_da.x.max())]
    dem_ys = [float(dem_da.y.min()), float(dem_da.y.max())]
    dem_xs_rg, dem_ys_rg = t_to_rg.transform(dem_xs, dem_ys)
    logger.info(
        f"DEM in radar grid CRS: x {min(dem_xs_rg):.0f}–{max(dem_xs_rg):.0f}, "
        f"y {min(dem_ys_rg):.0f}–{max(dem_ys_rg):.0f}"
    )

    x_overlap = max(dem_xs_rg) > x_rg.min() and min(dem_xs_rg) < x_rg.max()
    y_overlap = max(dem_ys_rg) > y_rg.min() and min(dem_ys_rg) < y_rg.max()
    if not (x_overlap and y_overlap):
        raise ValueError(
            "DEM and radar grid do not overlap! Check that GSLC covers your area."
        )

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

    # Process in parallel
    logger.info(f"Interpolating geometry data with {n_workers} workers")
    with mp.get_context("fork").Pool(processes=n_workers) as pool:
        for sl, results in tqdm(
            pool.imap(_interp_chunk, tasks),
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

    # Save all geometry layers at DEM (full) resolution
    logger.info("Saving geometry layers at full resolution (DEM grid)")

    # Save incidence angle
    logger.info(f"Saving incidence angle to {incidence_path}")
    inc_da = dem_da.copy(data=inc_surface).rio.write_crs(dem_crs)
    inc_da.rio.to_raster(incidence_path, compress="deflate", dtype="float32")
    logger.info(
        f"  Incidence angle range: {np.nanmin(inc_surface):.2f}–{np.nanmax(inc_surface):.2f} deg"
    )

    # Save LOS east component
    logger.info(f"Saving LOS east to {los_east_path}")
    los_east_da = dem_da.copy(data=los_east_surf).rio.write_crs(dem_crs)
    los_east_da.rio.to_raster(los_east_path, compress="deflate", dtype="float32")
    logger.info(
        f"  LOS east range: {np.nanmin(los_east_surf):.3f}–{np.nanmax(los_east_surf):.3f}"
    )

    # Save LOS north component
    logger.info(f"Saving LOS north to {los_north_path}")
    los_north_da = dem_da.copy(data=los_north_surf).rio.write_crs(dem_crs)
    los_north_da.rio.to_raster(los_north_path, compress="deflate", dtype="float32")
    logger.info(
        f"  LOS north range: {np.nanmin(los_north_surf):.3f}–{np.nanmax(los_north_surf):.3f}"
    )

    # Compute and save layover/shadow mask
    logger.info("Computing layover/shadow mask")
    layover_shadow_mask = _compute_layover_shadow_mask(
        inc_surface, dem_matched=dem_val
    )

    # Save as uint8: 0=bad (layover/shadow), 1=good
    layover_shadow_da = xr.DataArray(
        layover_shadow_mask.astype(np.uint8),
        dims=dem_da.dims,
        coords=dem_da.coords,
    ).rio.write_crs(dem_crs)
    layover_shadow_da.rio.write_nodata(255, inplace=True)
    layover_shadow_da.rio.to_raster(
        layover_shadow_path, compress="deflate", dtype="uint8"
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
    ref = rxr.open_rasterio(reference_raster, masked=True).squeeze()

    outputs = {}

    # Downsample incidence angle
    inc_out = output_dir / "incidence_angle_downsampled.tif"
    inc_da = rxr.open_rasterio(incidence_angle_path, masked=True).squeeze()
    inc_matched = inc_da.rio.reproject_match(ref, resampling=Resampling.bilinear)
    inc_matched.rio.to_raster(inc_out, compress="deflate", dtype="float32")
    outputs["incidence_angle"] = inc_out
    logger.info(f"Saved downsampled incidence angle to {inc_out}")

    # Downsample LOS east
    los_e_out = output_dir / "los_east_downsampled.tif"
    los_e_da = rxr.open_rasterio(los_east_path, masked=True).squeeze()
    los_e_matched = los_e_da.rio.reproject_match(ref, resampling=Resampling.bilinear)
    los_e_matched.rio.to_raster(los_e_out, compress="deflate", dtype="float32")
    outputs["los_east"] = los_e_out
    logger.info(f"Saved downsampled LOS east to {los_e_out}")

    # Downsample LOS north
    los_n_out = output_dir / "los_north_downsampled.tif"
    los_n_da = rxr.open_rasterio(los_north_path, masked=True).squeeze()
    los_n_matched = los_n_da.rio.reproject_match(ref, resampling=Resampling.bilinear)
    los_n_matched.rio.to_raster(los_n_out, compress="deflate", dtype="float32")
    outputs["los_north"] = los_n_out
    logger.info(f"Saved downsampled LOS north to {los_n_out}")

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
