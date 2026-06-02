from __future__ import annotations

import functools
import logging
import shutil
from collections.abc import Sequence
from multiprocessing import get_context
from os import fspath
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
from shapely.geometry import LinearRing, MultiPolygon, Polygon
from tqdm.contrib.concurrent import thread_map

try:
    from osgeo import gdal, osr

    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False
    gdal = None
    osr = None

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


def _unmangle_url(s: str) -> str:
    """Recover a remote URL that was damaged by passing through ``pathlib.Path``.

    Remote GSLC inputs flow through ``List[Path]`` and dolphin's path
    resolution, which mangles them two ways:

    * ``Path("https://host/x")`` collapses the ``//`` -> ``"https:/host/x"``;
    * because the result no longer starts with ``/``, it is then treated as a
      *relative* path and joined onto a base dir, e.g.
      ``"/scratch/data/https:/host/x"``.

    This finds an embedded ``http(s):/`` / ``s3:/`` scheme anywhere in the
    string, strips everything before it, and restores the ``//``. Plain local
    paths (no scheme) are returned unchanged.
    """
    s = str(s)
    for scheme in ("https", "http", "s3"):
        marker = f"{scheme}:/"
        idx = s.find(marker)
        if idx != -1:
            rest = s[idx + len(marker) :].lstrip("/")
            return f"{scheme}://{rest}"
    return s


def _is_remote_url(s: str) -> bool:
    return s.startswith(("http://", "https://", "s3://"))


@functools.lru_cache(maxsize=256)
def _read_nisar_bbox_streamed_cached(
    url: str, freq: str
) -> tuple[int, tuple[float, float, float, float]] | None:
    """Read EPSG + bounds from a remote NISAR GSLC by streaming it.

    Uses opera-utils' ``open_h5`` (Earthdata/S3 auth via fsspec) so authenticated
    ``https://`` / ``s3://`` URLs work where GDAL's HDF5 driver cannot open them.
    """
    try:
        from opera_utils._remote import open_h5

        with open_h5(url) as src:
            grp = src[f"science/LSAR/GSLC/grids/frequency{freq}"]
            epsg = int(grp["projection"][()])
            x = grp["xCoordinates"][()]
            y = grp["yCoordinates"][()]
            x_spacing = float(grp["xCoordinateSpacing"][()])
            y_spacing = float(grp["yCoordinateSpacing"][()])
    except Exception as e:  # noqa: BLE001
        logger.debug(f"_read_nisar_bbox_streamed_cached failed for {url}: {e}")
        return None

    hx = abs(x_spacing) / 2.0
    hy = abs(y_spacing) / 2.0
    bounds = (
        float(x.min()) - hx,
        float(y.min()) - hy,
        float(x.max()) + hx,
        float(y.max()) + hy,
    )
    return int(epsg), bounds


def get_nisar_frame_bbox(
    cslc_file: Path,
    frequency: str = "frequencyA",
    polarization: str = "HH",  # noqa: ARG001
) -> tuple[int, Bbox]:
    """Return ``(epsg, Bbox(left, bottom, right, top))`` for a CSLC file.

    For NISAR GSLC (``.h5`` / ``.hdf5``) this uses GDAL's multidim API, so
    it works for local paths and ``/vsis3/...`` URLs alike. The
    ``polarization`` argument is unused (the projected grid is shared
    across polarizations within a frequency) and is kept for API
    compatibility.
    """
    path = _unmangle_url(fspath(cslc_file))
    ext = Path(path).suffix.lower()

    if ext in {".h5", ".hdf5"}:
        # Accept either "A"/"B" or "frequencyA"/"frequencyB"
        freq_short = frequency.removeprefix("frequency") or "A"
        if _is_remote_url(path):
            # Authenticated remote URL: GDAL's HDF5 driver can't open it; stream.
            result = _read_nisar_bbox_streamed_cached(path, freq_short)
        else:
            result = _read_nisar_bbox_multidim_cached(path, freq_short)
        if result is None:
            msg = f"Could not read EPSG/bbox from {path}"
            raise RuntimeError(msg)
        epsg, bounds = result
        return epsg, Bbox(*bounds)

    # Alternative format (non-NISAR HDF5) — local h5py path, unchanged.
    import h5py

    with h5py.File(path, "r") as src:
        epsg = int(src["data"]["spatial_ref"][()])
        data = src["data"]
        bounds = (
            float(data["x"][()].min()),
            float(data["y"][()].min()),
            float(data["x"][()].max()),
            float(data["y"][()].max()),
        )

    return epsg, Bbox(*bounds)

    # if cslc_file.suffix in {".h5", ".hdf5"}:
    #     import h5py

    #     # Read CRS and bounds directly from NISAR HDF5 metadata
    #     with h5py.File(cslc_file, "r") as h5f:
    #         grid_group = h5f[f"science/LSAR/GSLC/grids/{frequency}"]
    #         epsg = int(grid_group["projection"][()])

    #         x_coords = grid_group["xCoordinates"][:]
    #         y_coords = grid_group["yCoordinates"][:]
    #         x_spacing = float(grid_group["xCoordinateSpacing"][()])
    #         y_spacing = float(grid_group["yCoordinateSpacing"][()])

    #         # Compute bounds (left, bottom, right, top)
    #         bounds = (
    #             float(x_coords.min()) - abs(x_spacing) / 2,
    #             float(y_coords.min()) - abs(y_spacing) / 2,
    #             float(x_coords.max()) + abs(x_spacing) / 2,
    #             float(y_coords.max()) + abs(y_spacing) / 2,
    #         )
    # else:
    #     import h5py

    #     # Alternative format handling (non-NISAR HDF5)
    #     with h5py.File(cslc_file, "r") as src:
    #         epsg = src["data"]["spatial_ref"][()]
    #         data = src["data"]

    #         bounds = (
    #             data["x"][()].min(),
    #             data["y"][()].min(),
    #             data["x"][()].max(),
    #             data["y"][()].max(),
    #         )

    # return epsg, Bbox(*bounds)


@functools.lru_cache(maxsize=256)
def _read_nisar_bbox_multidim_cached(
    path: str, freq: str
) -> tuple[int, tuple[float, float, float, float]] | None:
    ds = grp = None
    try:
        ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER)
        if ds is None:
            return None
        grp = ds.GetRootGroup()
        for name in ("science", "LSAR", "GSLC", "grids", f"frequency{freq}"):
            grp = grp.OpenGroup(name)
            if grp is None:
                return None

        f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        x = grp.OpenMDArray("xCoordinates").ReadAsArray(buffer_datatype=f64)
        y = grp.OpenMDArray("yCoordinates").ReadAsArray(buffer_datatype=f64)

        # Authoritative spacings (match h5py code); fall back to diffs.
        x_spacing = _read_scalar_or(grp, "xCoordinateSpacing", x)
        y_spacing = _read_scalar_or(grp, "yCoordinateSpacing", y)

        epsg = _epsg_from_projection_mdarray(grp.OpenMDArray("projection"))
    except Exception as e:
        logger.debug(f"_read_nisar_bbox_multidim_cached failed for {path}: {e}")
        return None
    finally:
        grp = ds = None

    if (
        x is None
        or y is None
        or x.size == 0
        or y.size == 0
        or epsg is None
        or x_spacing is None
        or y_spacing is None
    ):
        return None

    hx = abs(x_spacing) / 2.0
    hy = abs(y_spacing) / 2.0
    bounds = (
        float(x.min()) - hx,
        float(y.min()) - hy,
        float(x.max()) + hx,
        float(y.max()) + hy,
    )
    return int(epsg), bounds


def _read_scalar_or(grp, name: str, fallback_arr) -> float | None:
    """Read a scalar MDArray as float; fall back to step of ``fallback_arr``."""
    try:
        return float(grp.OpenMDArray(name).ReadAsArray().item())
    except Exception:
        try:
            if fallback_arr is not None and fallback_arr.size > 1:
                return float(fallback_arr[1] - fallback_arr[0])
        except Exception:
            pass
    return None


def _read_nisar_projection_wkt(proj_ar) -> str | None:
    """Return the projection WKT from a NISAR ``projection`` MDArray's attributes.

    NISAR GSLC ``projection`` datasets carry the CRS WKT in a ``spatial_ref``
    (or ``crs_wkt``) attribute. Returns ``None`` if no WKT attribute is found.
    Inlined here (previously ``opera_utils._cslc._read_nisar_projection_wkt``,
    since removed from opera-utils).
    """
    if proj_ar is None:
        return None
    for name in ("spatial_ref", "crs_wkt", "projection_wkt"):
        try:
            attr = proj_ar.GetAttribute(name)
        except Exception:
            attr = None
        if attr is None:
            continue
        try:
            value = attr.Read()
        except Exception:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8", "replace")
        if value:
            return str(value)
    return None


def _epsg_from_projection_mdarray(proj_ar) -> int | None:
    """Pull an integer EPSG code from a NISAR ``projection`` MDArray.

    NISAR stores the EPSG as the scalar value of the ``projection``
    variable (``projection[()]`` in h5py); fall back to an
    ``epsg_code`` attribute or a WKT-derived authority code.
    """
    # Primary: scalar data value — matches `grid_group["projection"][()]`.
    try:
        return int(proj_ar.ReadAsArray().item())
    except Exception:
        pass
    # Fallback 1: `epsg_code` attribute.
    try:
        attr = proj_ar.GetAttribute("epsg_code")
        if attr is not None:
            v = attr.Read()
            if isinstance(v, (list, tuple)):
                v = v[0]
            return int(v)
    except Exception:
        pass
    # Fallback 2: derive from WKT (uses your existing helper).
    wkt = _read_nisar_projection_wkt(proj_ar)
    if wkt:
        srs = osr.SpatialReference()
        if srs.ImportFromWkt(wkt) == 0:
            code = srs.GetAuthorityCode(None)
            if code:
                try:
                    return int(code)
                except ValueError:
                    return None
    return None


def _frequency_to_wavelength(frequency: str, gslc_file: Filename) -> float:
    dset = f"/science/LSAR/GSLC/grids/{frequency}/centerFrequency"
    url = _unmangle_url(fspath(gslc_file))
    if _is_remote_url(url):
        # Stream centerFrequency from the authenticated remote URL.
        from opera_utils._remote import open_h5

        with open_h5(url) as src:
            center_frequency = float(src[dset][()])
    else:
        center_frequency = _get_dset_and_attrs(filename=gslc_file, dset_name=dset)[0]
    wavelength = SPEED_OF_LIGHT / center_frequency
    return wavelength
