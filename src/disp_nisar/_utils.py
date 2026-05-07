from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import shapely.ops
from dolphin import PathOrStr, io
from dolphin._types import Bbox, Filename
from dolphin.constants import SPEED_OF_LIGHT
from dolphin.interferogram import estimate_correlation_from_phase
from dolphin.unwrap import grow_conncomp_snaphu
from dolphin.utils import full_suffix
from dolphin.workflows.config import UnwrapOptions
from opera_utils._cslc import _get_dset_and_attrs
from osgeo import gdal
from shapely.geometry import LinearRing, MultiPolygon, Polygon
from tqdm.contrib.concurrent import thread_map

logger = logging.getLogger(__name__)

gdal.UseExceptions()


def _update_snaphu_conncomps(
    timeseries_paths: Sequence[Path],
    stitched_cor_paths: Sequence[Path],
    mask_filename: PathOrStr,
    unwrap_options: UnwrapOptions,
    nlooks: int,
    max_workers: int = 2,
) -> list[Path]:
    """Recompute connected components from SNAPHU after a timeseries inversion.

    `timeseries_paths` contains the post-inversion rasters, one per secondary date.

    Parameters
    ----------
    timeseries_paths : list[Path]
        list of paths to the timeseries files.
    stitched_cor_paths : list[Path]
        list of paths to the pseuedo-correlation rasters.
    mask_filename : PathOrStr
        Path to a binary mask matching shape of `timeseries_paths`.
    unwrap_options : [dolphin.workflows.config.UnwrapOptions][]
        Configuration object containing unwrapping options.
    nlooks : int
        Effective number of looks used to make correlation.
    max_workers : int
        Number of parallel files to process.
        Default is 2.

    Returns
    -------
    list[Path]
        list of updated connected component paths.

    """
    args_list = [
        (idx, unw_f, cor_f, nlooks, mask_filename, unwrap_options)
        for idx, (unw_f, cor_f) in enumerate(zip(timeseries_paths, stitched_cor_paths))
    ]

    mp_context = get_context("spawn")
    with mp_context.Pool(max_workers) as pool:
        return list(pool.map(_regrow, args_list))


def _regrow(args: tuple[int, Path, Path, int, PathOrStr, UnwrapOptions]) -> Path:
    scratch_idx, unw_f, cor_f, nlooks, mask_filename, unwrap_options = args
    new_path = grow_conncomp_snaphu(
        unw_filename=unw_f,
        corr_filename=cor_f,
        nlooks=nlooks,
        mask_filename=mask_filename,
        cost=unwrap_options.snaphu_options.cost,
        scratchdir=unwrap_options._directory / f"scratch{scratch_idx}",
    )
    return new_path


def _update_spurt_conncomps(
    timeseries_paths: Sequence[Path],
    template_conncomp_path: Path,
) -> list[Path]:
    """Recompute connected components from spurt after a timeseries inversion.

    Since spurt uses one file computed from `ndimage.label`, we just need to
    rename an example to be the same as the timeseries rasters.

    Parameters
    ----------
    timeseries_paths : list[Path]
        list of paths to the timeseries files.
    template_conncomp_path : Path
        One connected component paths from the spurt unwrapping.
        Only one is needed while spurt uses only a single mask for pixel selection.

    Returns
    -------
    list[Path]
        list of updated connected component paths.

    """
    new_conncomp_paths: list[Path] = []
    for ts_p in timeseries_paths:
        new_name = template_conncomp_path.parent / str(ts_p.name).replace(
            full_suffix(ts_p), full_suffix(template_conncomp_path)
        )
        try:
            shutil.copy(template_conncomp_path, new_name)
        except shutil.SameFileError:
            pass
        new_conncomp_paths.append(new_name)
    return new_conncomp_paths


def _create_correlation_images(
    ts_filenames: Sequence[PathOrStr],
    wavelength: float,
    window_size: tuple[int, int] = (11, 11),
    keep_bits: int = 8,
    num_workers: int = 3,
) -> list[Path]:
    path_tuples: list[tuple[Path, Path]] = []
    output_paths: list[Path] = []
    for fn in ts_filenames:
        ifg_path = Path(fn)
        cor_path = ifg_path.with_suffix(".cor.tif")
        output_paths.append(cor_path)
        if cor_path.exists():
            logger.info(f"Skipping existing interferometric correlation for {ifg_path}")
            continue
        path_tuples.append((ifg_path, cor_path))

    def process_ifg(args):
        ifg_path, cor_path = args
        logger.debug(f"Estimating correlation for {ifg_path}, writing to {cor_path}")
        disp = io.load_gdal(ifg_path)

        METERS_TO_RADIANS = (-4 * np.pi) / wavelength
        disp_rad = disp * METERS_TO_RADIANS

        cor = estimate_correlation_from_phase(disp_rad, window_size=window_size)
        if keep_bits:
            io.round_mantissa(cor, keep_bits=keep_bits)

        io.write_arr(
            arr=cor,
            output_name=cor_path,
            like_filename=ifg_path,
            driver="GTiff",
        )

    thread_map(
        process_ifg,
        path_tuples,
        max_workers=num_workers,
        desc="Estimating correlations",
    )

    return output_paths


def extract_footprint(raster_path: PathOrStr, simplify_tolerance: float = 0.01) -> str:
    """Extract a simplified footprint from a raster file.

    This function opens a raster file, extracts its footprint, simplifies it,
    and returns the a Polygon from the exterior ring as a WKT string.

    Parameters
    ----------
    raster_path : str
        Path to the input raster file.
    simplify_tolerance : float, optional
        Tolerance for simplification of the footprint geometry.
        Default is 0.01.

    Returns
    -------
    str
        WKT string representing the simplified exterior footprint
        in EPSG:4326 (lat/lon) coordinates.

    Notes
    -----
    This function uses GDAL to open the raster and extract the footprint,
    and Shapely to process the geometry.

    """
    from os import fspath

    import shapely
    from osgeo import gdal

    # Extract the footprint as WKT string (don't save)
    wkt = gdal.Footprint(
        None,
        fspath(raster_path),
        format="WKT",
        dstSRS="EPSG:4326",
        simplify=simplify_tolerance,
    )

    # Convert WKT to Shapely geometry, extract exterior, and convert back to Polygon WKT
    in_multi = shapely.from_wkt(wkt)

    # This may have holes; get the exterior
    # Largest polygon should be first in MultiPolygon returned by GDAL
    footprint = shapely.Polygon(in_multi.geoms[0].exterior)
    # Split on antimeridian and return the WKT string
    return split_on_antimeridian(footprint).wkt


def split_on_antimeridian(polygon: Polygon) -> MultiPolygon:
    """Split `polygon` if it crosses the antimeridian (180°).

    Source:
    https://github.com/nasa/opera-sds-pcm/blob/a5a3db25be462e7955e5de06d6f9d1d8236a1ef2/util/geo_util.py#L265
    (where it is `check_dateline`, as it is in isce3)

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Input polygon.

    Returns
    -------
    MultiPolygon
        A MultiPolygon containing 1 or 2 `.geoms`:
        The input polygon if it didn't cross the antimeridian, or
        two polygons otherwise (one on either side of the antimeridian).

    """
    x_min, _, x_max, _ = polygon.bounds

    # Check antimeridian crossing
    if (x_max - x_min > 180.0) or (x_min <= 180.0 <= x_max):
        antimeridian = shapely.wkt.loads("LINESTRING( 180.0 -90.0, 180.0 90.0)")

        # build new polygon with all longitudes between 0 and 360
        x, y = polygon.exterior.coords.xy
        new_x = (k + (k <= 0.0) * 360 for k in x)
        new_ring = LinearRing(zip(new_x, y))

        # Split input polygon
        # (https://gis.stackexchange.com/questions/232771/splitting-polygon-by-linestring-in-geodjango_)
        merged_lines = shapely.ops.linemerge([antimeridian, new_ring])
        border_lines = shapely.ops.unary_union(merged_lines)
        decomp = shapely.ops.polygonize(border_lines)

        polys = list(decomp)

        for polygon_count in range(len(polys)):
            x, y = polys[polygon_count].exterior.coords.xy
            # if there are no longitude values above 180, continue
            if not any(k > 180 for k in x):
                continue

            # otherwise, wrap longitude values down by 360 degrees
            x_wrapped_minus_360 = np.asarray(x) - 360
            polys[polygon_count] = Polygon(zip(x_wrapped_minus_360, y))

    else:
        # If antimeridian is not crossed, treat input polygon as list
        polys = [polygon]

    return MultiPolygon(polys)


def _convert_meters_to_radians(
    timeseries_paths: Sequence[Path], wavelength: float
) -> list[Path]:
    """Copy over .tif, rescaling units from meters to radians."""
    output_files: list[Path] = []
    METERS_TO_RADIANS = (-4 * np.pi) / wavelength

    for in_path in timeseries_paths:
        out_path = in_path.with_suffix(".radians.tif")
        io.write_arr(
            arr=METERS_TO_RADIANS * io.load_gdal(in_path),
            like_filename=in_path,
            output_name=out_path,
        )
        output_files.append(out_path)
    return output_files


def _extract_wkt(*candidates) -> str:
    """Pull a WKT string out of whatever GDAL's multidim API hands back.

    Handles Python str / bytes / bytearray / nested lists / numpy object
    arrays / and attribute dicts keyed by things like `spatial_ref` or `wkt`.
    """
    _WKT_PREFIXES = ("PROJCS", "GEOGCS", "PROJCRS", "GEOGCRS", "COMPD_CS", "LOCAL_CS")

    def _from_value(v):
        if v is None:
            return ""
        if isinstance(v, str):
            s = v.strip().lstrip("\x00").rstrip("\x00").strip()
            return s if s.upper().startswith(_WKT_PREFIXES) else ""
        if isinstance(v, (bytes, bytearray)):
            try:
                return _from_value(bytes(v).decode("utf-8", errors="replace"))
            except Exception:
                return ""
        if isinstance(v, np.ndarray):
            # flat-iter handles 0-d, 1-d, and object dtype with one element
            for item in v.ravel():
                got = _from_value(item)
                if got:
                    return got
            return ""
        if isinstance(v, (list, tuple)):
            for item in v:
                got = _from_value(item)
                if got:
                    return got
            return ""
        return ""

    for c in candidates:
        if isinstance(c, dict):
            for key in ("spatial_ref", "crs_wkt", "wkt", "projection", "srs"):
                if key in c:
                    got = _from_value(c[key])
                    if got:
                        return got
            for v in c.values():
                got = _from_value(v)
                if got:
                    return got
        else:
            got = _from_value(c)
            if got:
                return got
    return ""


def _read_nisar_grid_mdarrays(
    path: str, frequency: str
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Read x/y coordinates, spacings, and EPSG from a NISAR GSLC via multidim API.

    The `projection` MDArray holds the full WKT for the grid; EPSG is parsed
    from that rather than read as a scalar.
    """
    ds = ds_grp = None
    try:
        ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER)
        if ds is None:
            msg = f"Could not open {path} via multidim API"
            raise ValueError(msg)
        grp = ds.GetRootGroup()
        for name in ("science", "LSAR", "GSLC", "grids", frequency):
            grp = grp.OpenGroup(name)
            if grp is None:
                msg = f"Group {name!r} missing under {path}"
                raise ValueError(msg)
        ds_grp = grp

        f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        x_coords = ds_grp.OpenMDArray("xCoordinates").ReadAsArray(buffer_datatype=f64)
        y_coords = ds_grp.OpenMDArray("yCoordinates").ReadAsArray(buffer_datatype=f64)
        x_spacing = float(
            ds_grp.OpenMDArray("xCoordinateSpacing")
            .ReadAsArray(buffer_datatype=f64)
            .item()
        )
        y_spacing = float(
            ds_grp.OpenMDArray("yCoordinateSpacing")
            .ReadAsArray(buffer_datatype=f64)
            .item()
        )
        proj_mdar = ds_grp.OpenMDArray("projection")
        epsg = None
        try:
            a = proj_mdar.GetAttribute("epsg_code")
            if a is not None:
                val = a.Read()
                if isinstance(val, (list, tuple)) and val:
                    val = val[0]
                epsg = int(val)
        except Exception:
            epsg = None
        if epsg is None:
            arr = proj_mdar.ReadAsArray()
            if arr is not None and arr.size == 1:
                epsg = int(arr.item())
    finally:
        ds_grp = ds = None

    if epsg is None:
        msg = (
            "Could not read EPSG code from"
            f" {path}:/science/LSAR/GSLC/grids/{frequency}/projection"
        )
        raise ValueError(msg)

    return x_coords, y_coords, x_spacing, y_spacing, epsg


def get_nisar_frame_bbox(
    cslc_file: Filename,
    frequency: str = "frequencyA",
    polarization: str = "HH",  # noqa: ARG001
) -> tuple[int, Bbox]:
    """Extract the EPSG code and bounding box from a NISAR CSLC file.

    Parameters
    ----------
    cslc_file : Filename
        path to the NISAR CSLC file (.h5 or .hdf5). May be a VSI path
        (e.g. /vsis3/...), in which case metadata is read via GDAL's
        multidim API.
    frequency : str
        Frequency band to use (default: "frequencyA")
    polarization : str
        Polarization to use (default: "HH")

    Returns
    -------
    tuple[int, Bbox]
        (EPSG code, Bounding box)

    Raises
    ------
    ValueError: If required metadata is missing

    """
    path_str = str(cslc_file)
    suffix = Path(path_str).suffix

    if path_str.startswith("/vsi"):
        x_coords, y_coords, x_spacing, y_spacing, epsg = _read_nisar_grid_mdarrays(
            path_str, frequency
        )
        bounds = (
            float(x_coords.min()) - abs(x_spacing) / 2,
            float(y_coords.min()) - abs(y_spacing) / 2,
            float(x_coords.max()) + abs(x_spacing) / 2,
            float(y_coords.max()) + abs(y_spacing) / 2,
        )
    elif suffix in {".h5", ".hdf5"}:
        import h5py

        # Read CRS and bounds directly from NISAR HDF5 metadata
        with h5py.File(cslc_file, "r") as h5f:
            grid_group = h5f[f"science/LSAR/GSLC/grids/{frequency}"]
            epsg = int(grid_group["projection"][()])

            x_coords = grid_group["xCoordinates"][:]
            y_coords = grid_group["yCoordinates"][:]
            x_spacing = float(grid_group["xCoordinateSpacing"][()])
            y_spacing = float(grid_group["yCoordinateSpacing"][()])

            # Compute bounds (left, bottom, right, top)
            bounds = (
                float(x_coords.min()) - abs(x_spacing) / 2,
                float(y_coords.min()) - abs(y_spacing) / 2,
                float(x_coords.max()) + abs(x_spacing) / 2,
                float(y_coords.max()) + abs(y_spacing) / 2,
            )
    else:
        import h5py

        # Alternative format handling (non-NISAR HDF5)
        with h5py.File(cslc_file, "r") as src:
            epsg = src["data"]["spatial_ref"][()]
            data = src["data"]

            bounds = (
                data["x"][()].min(),
                data["y"][()].min(),
                data["x"][()].max(),
                data["y"][()].max(),
            )

    return epsg, Bbox(*bounds)


def _frequency_to_wavelength(frequency: str, gslc_file: Filename) -> float:
    dset = f"/science/LSAR/GSLC/grids/{frequency}/centerFrequency"
    center_frequency = _get_dset_and_attrs(filename=gslc_file, dset_name=dset)[0]
    wavelength = SPEED_OF_LIGHT / center_frequency
    return wavelength
