"""Azimuth-block orchestration for disp-nisar.

Splits a NISAR frame into N azimuth blocks, runs dolphin's phase-linking stage
per block with `output_options.bounds` narrowed to each block's read window,
then assembles the per-block outputs into full-frame scratch rasters and runs
unwrapping + timeseries once on the full frame.

Halos on internal boundaries come from `phase_linking.half_window`
(`overlap = max(half_window.x, half_window.y)`), so each block's central rows
are computed with the same spatial neighborhood they would see in a monolithic
run. Because the written central rows are disjoint and abutting, no weighted
stitching is needed.

Three execution modes are driven by `azimuth_blocks.block_index`:

* ``None`` (default) — run every block in this process (sequential or via a
  `ProcessPoolExecutor`) then assemble and continue to unwrap/timeseries.
* ``0 <= k < num_blocks`` — run only block ``k`` and write its per-block
  outputs to ``shard_dir/block_k/``. Used by AWS Batch workers.
* ``-1`` — skip phase linking, assume shards already exist in ``shard_dir``,
  then assemble and continue. Used by the batch finalize job.
"""

from __future__ import annotations

import copy
import functools
import gc
import logging
import multiprocessing as mp
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

from dolphin import interferogram, io
from dolphin._types import Bbox, Filename
from dolphin.workflows import wrapped_phase
from dolphin.workflows.config import DisplacementWorkflow
from dolphin.workflows.displacement import OutputPaths
from osgeo import gdal, osr

gdal.UseExceptions()

logger = logging.getLogger(__name__)

# Paths pointing at remote storage get materialized locally before a block
# runs — see `_stage_inputs_for_block`. Anything not starting with one of
# these prefixes is treated as a local file and left untouched.
_REMOTE_PATH_PREFIXES = ("/vsis3/", "/vsicurl/", "s3://", "http://", "https://")


class BlockWindow(NamedTuple):
    """One azimuth block's row windows.

    Rows are indexed from the top of the full frame (row 0 = ymax in UTM),
    matching GDAL's raster coordinate convention.
    """

    block_index: int
    # Rows the block reads from the GSLCs (central + halo)
    read_start: int
    read_stop: int
    # Rows the block writes into the assembled full-frame raster
    write_start: int
    write_stop: int

    @property
    def read_height(self) -> int:
        return self.read_stop - self.read_start

    @property
    def write_height(self) -> int:
        return self.write_stop - self.write_start

    @property
    def write_offset_in_block(self) -> int:
        """Offset of the central region within the block's own raster."""
        return self.write_start - self.read_start


@dataclass
class FullFrameGrid:
    """Geometry of the full-frame output grid, shared by every assembled raster."""

    bounds: Bbox
    epsg: int
    x_res: float  # positive pixel size (m)
    y_res: float  # positive pixel size (m)
    rows: int
    cols: int

    @property
    def geotransform(self) -> tuple[float, float, float, float, float, float]:
        # North-up: top-left origin, negative y pixel height
        return (
            float(self.bounds.left),
            float(self.x_res),
            0.0,
            float(self.bounds.top),
            0.0,
            -float(self.y_res),
        )


def compute_block_windows(
    total_rows: int, num_blocks: int, overlap: int
) -> list[BlockWindow]:
    """Partition ``total_rows`` into ``num_blocks`` central windows with halos.

    For ``num_blocks=5``, ``total_rows=500``, ``overlap=7`` the result is::

        index | read_start read_stop | write_start write_stop
        ------+-----------------------+------------------------
          0   |      0         107   |      0         100
          1   |     93         207   |    100         200
          2   |    193         307   |    200         300
          3   |    293         407   |    300         400
          4   |    393         500   |    400         500

    Inner blocks read an ``overlap`` halo on each side; the first/last blocks
    only have a halo on their interior side.
    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if total_rows < num_blocks:
        raise ValueError(
            f"total_rows ({total_rows}) must be >= num_blocks ({num_blocks})"
        )
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")

    # Use ceil so the last block absorbs any remainder; every central region
    # still has height >= 1.
    block_rows = -(-total_rows // num_blocks)
    windows: list[BlockWindow] = []
    for k in range(num_blocks):
        write_start = k * block_rows
        write_stop = min((k + 1) * block_rows, total_rows)
        if write_start >= total_rows:
            # Fewer than num_blocks blocks are actually needed; skip empties.
            break
        read_start = 0 if k == 0 else max(0, write_start - overlap)
        read_stop = (
            total_rows if k == num_blocks - 1 else min(total_rows, write_stop + overlap)
        )
        windows.append(
            BlockWindow(
                block_index=k,
                read_start=read_start,
                read_stop=read_stop,
                write_start=write_start,
                write_stop=write_stop,
            )
        )
    return windows


def block_bounds(frame: FullFrameGrid, block: BlockWindow) -> Bbox:
    """Translate a block's row read-window to projected bounds.

    Transforms pixel/row indices to coordinates in the projection specified by
    ``frame.epsg``. The x range spans the full frame; only y is narrowed to the
    block's read window.

    Uses the frame's geotransform to convert from pixel coordinates (row, col)
    to projected coordinates (x, y):
        x_projected = geotransform[0] + col * geotransform[1]
        y_projected = geotransform[3] + row * geotransform[5]

    For north-up rasters, geotransform[5] is negative, so:
        y_projected = top - row * y_res

    Parameters
    ----------
    frame : FullFrameGrid
        Full frame grid with bounds in projected coordinates (EPSG: frame.epsg)
    block : BlockWindow
        Block window with row indices (0-based from top of frame)

    Returns
    -------
    Bbox
        Bounding box in projected coordinates (meters or degrees depending on EPSG)

    """
    # Get geotransform components
    gt = frame.geotransform
    x_origin = gt[0]  # left (westernmost X)
    x_pixel_size = gt[1]  # pixel width
    y_origin = gt[3]  # top (northernmost Y)
    y_pixel_size = gt[5]  # pixel height (negative for north-up)

    # Transform row indices to Y coordinates in the projection
    # For north-up rasters: y_pixel_size is negative, so we add row * y_pixel_size
    # which is equivalent to subtracting row * abs(y_pixel_size)
    ymax_block = y_origin + block.read_start * y_pixel_size
    ymin_block = y_origin + block.read_stop * y_pixel_size

    # X coordinates span the full frame (columns 0 to frame.cols)
    xmin_block = x_origin
    xmax_block = x_origin + frame.cols * x_pixel_size

    return Bbox(
        left=xmin_block,
        bottom=min(ymin_block, ymax_block),  # bottom is smaller Y value
        right=xmax_block,
        top=max(ymin_block, ymax_block),  # top is larger Y value
    )


def resolve_overlap(cfg: DisplacementWorkflow) -> int:
    """Halo size in rows, derived from phase_linking.half_window (max of x, y)."""
    hw = cfg.phase_linking.half_window
    return max(int(hw.y), int(hw.x))


def build_full_frame_grid(
    cfg: DisplacementWorkflow, x_res: float, y_res: float, epsg: int
) -> FullFrameGrid:
    """Build a `FullFrameGrid` from the workflow's full-frame bounds + pixel size.

    Prefer :func:`load_grid_from_nisar_gslc` for NISAR — it reads the raster
    grid directly from the GSLC so block windows agree with the grid used by
    ``make_nodata_mask`` and by staged GTiffs. This function is kept for
    non-NISAR callers / tests that construct a grid from bounds alone.
    """
    bounds = cfg.output_options.bounds
    if bounds is None:
        raise ValueError(
            "cfg.output_options.bounds must be set before building a full-frame grid"
        )
    full = Bbox(*bounds) if not isinstance(bounds, Bbox) else bounds
    width = full.right - full.left
    height = full.top - full.bottom
    cols = int(round(width / x_res))
    rows = int(round(height / y_res))
    return FullFrameGrid(
        bounds=full, epsg=epsg, x_res=x_res, y_res=y_res, rows=rows, cols=cols
    )


# def _get_nisar_geotransform(
#     h5_file: Filename, frequency: str = "frequencyA"
# ) -> tuple[float, float, float, float, float, float] | None:
#     """Read geotransform directly from NISAR HDF5 metadata.

#     GDAL's HDF5 driver returns an identity matrix for NISAR files, so we need
#     to read the coordinates directly from the multidimensional metadata.

#     Parameters
#     ----------
#     h5_file : Filename
#         Path to NISAR GSLC HDF5 file
#     frequency : str
#         Frequency band to use (default: "frequencyA")

#     Returns
#     -------
#     tuple | None
#         6-element geotransform (left, x_res, x_rot, top, y_rot, -y_res)
#         or None if metadata cannot be read

#     """
#     import h5py

#     try:
#         with h5py.File(h5_file, "r") as h5f:
#             grid_group = h5f[f"science/LSAR/GSLC/grids/{frequency}"]

#             x_coords = grid_group["xCoordinates"][:]
#             y_coords = grid_group["yCoordinates"][:]
#             x_spacing = float(grid_group["xCoordinateSpacing"][()])
#             y_spacing = float(grid_group["yCoordinateSpacing"][()])

#             # NISAR coordinates are pixel centers
#             # Geotransform origin is top-left corner
#             left = float(x_coords.min()) - abs(x_spacing) / 2
#             top = float(y_coords.max()) + abs(y_spacing) / 2

#             # Standard geotransform: (left, x_res, x_rot, top, y_rot, -y_res)
#             # Note: y_spacing is negative for north-up orientation
#             return (left, abs(x_spacing), 0.0, top, 0.0, -abs(y_spacing))

#     except Exception as e:
#         logger.debug(f"Could not read NISAR geotransform from {h5_file}: {e}")
#         return None


def load_grid_from_nisar_gslc(
    gslc_path: object, subdataset: str | None, epsg: int
) -> FullFrameGrid:
    """Return the authoritative native grid of a NISAR GSLC.

    Reads x/yCoordinates and x/yCoordinateSpacing via GDAL's multidim API so
    it works for both local paths and ``/vsis3/...`` URLs. The classic HDF5
    driver does not expose a geotransform for NISAR GSLCs.
    """
    from osgeo import gdal

    src_str = str(gslc_path)

    # Resolve frequency from the subdataset path (e.g. ".../frequencyA/HH").
    frequency = "frequencyA"
    if subdataset and "frequency" in subdataset.lower():
        for part in subdataset.split("/"):
            if part.startswith("frequency"):
                frequency = part
                break

    is_hdf5 = src_str.lower().endswith((".h5", ".hdf5"))

    gt: tuple[float, ...] | None = None
    rows = cols = None

    # Preferred path for NISAR GSLCs: multidim API. Works over /vsis3.
    if is_hdf5:
        info = _read_nisar_grid_multidim(src_str, frequency)
        if info is not None:
            gt, rows, cols = info
            logger.debug(f"Got NISAR grid via multidim API: gt={gt} ({cols}x{rows})")

    # Fallback for non-NISAR / non-HDF5 inputs (geotiffs, properly tagged
    # netCDFs, etc.). Won't recover NISAR GSLCs — those must go through
    # the multidim path above.
    if gt is None or tuple(gt) == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        logger.debug("Falling back to standard GDAL GetGeoTransform")
        if subdataset and src_str.lower().endswith((".h5", ".hdf5", ".nc")):
            uri = f'HDF5:"{src_str}"://{subdataset.lstrip("/")}'
        else:
            uri = src_str
        ds = gdal.Open(uri)
        if ds is None:
            raise RuntimeError(f"GDAL could not open NISAR GSLC at {uri}")
        try:
            gt = ds.GetGeoTransform()
            rows = ds.RasterYSize
            cols = ds.RasterXSize
        finally:
            ds = None

    if gt is None or tuple(gt) == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        raise RuntimeError(
            f"Could not get valid geotransform from {src_str}. "
            "Got identity matrix, which means the file is not properly georeferenced."
        )
    if rows is None or cols is None:
        raise RuntimeError(f"Could not determine raster size for {src_str}")

    left = gt[0]
    x_res = gt[1]
    top = gt[3]
    y_res = abs(gt[5])
    right = left + cols * x_res
    bottom = top - rows * y_res

    logger.info(
        f"Loaded NISAR GSLC grid: {cols}x{rows} pixels, "
        f"bounds: ({left:.2f}, {bottom:.2f}, {right:.2f}, {top:.2f}), "
        f"resolution: {x_res:.2f} x {y_res:.2f}"
    )

    return FullFrameGrid(
        bounds=Bbox(left, bottom, right, top),
        epsg=int(epsg),
        x_res=x_res,
        y_res=y_res,
        rows=rows,
        cols=cols,
    )


@functools.lru_cache(maxsize=256)
def _read_nisar_grid_multidim(
    path: str, frequency: str
) -> tuple[tuple[float, float, float, float, float, float], int, int] | None:
    """Return (geotransform, rows, cols) for a NISAR GSLC.

    Combines pixel-center ``x/yCoordinates`` with the signed
    ``x/yCoordinateSpacing`` scalars to produce a GDAL-convention
    (outer-corner) geotransform. Works for local paths and ``/vsis3``.
    """
    from osgeo import gdal

    # Accept "A"/"B" or "frequencyA"/"frequencyB".
    freq_short = frequency.removeprefix("frequency") or "A"

    ds = grp = None
    try:
        ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER)
        if ds is None:
            return None
        grp = ds.GetRootGroup()
        for name in ("science", "LSAR", "GSLC", "grids", f"frequency{freq_short}"):
            grp = grp.OpenGroup(name)
            if grp is None:
                return None

        f64 = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        x = grp.OpenMDArray("xCoordinates").ReadAsArray(buffer_datatype=f64)
        y = grp.OpenMDArray("yCoordinates").ReadAsArray(buffer_datatype=f64)

        # Authoritative spacings (signed); fall back to diffs if absent.
        try:
            dx = float(grp.OpenMDArray("xCoordinateSpacing").ReadAsArray().item())
        except Exception:
            dx = float(x[1] - x[0]) if x is not None and x.size > 1 else 0.0
        try:
            dy = float(grp.OpenMDArray("yCoordinateSpacing").ReadAsArray().item())
        except Exception:
            dy = float(y[1] - y[0]) if y is not None and y.size > 1 else 0.0
    except Exception as e:
        logger.debug(f"_read_nisar_grid_multidim failed for {path}: {e}")
        return None
    finally:
        grp = ds = None

    if x is None or y is None or x.size < 1 or y.size < 1 or dx == 0.0 or dy == 0.0:
        return None

    # xCoordinates/yCoordinates are pixel CENTERS; GDAL's geotransform
    # origin is the OUTER CORNER of the top-left pixel.
    gt = (
        float(x[0]) - dx / 2.0,
        dx,
        0.0,
        float(y[0]) - dy / 2.0,
        0.0,
        dy,
    )
    return gt, int(y.size), int(x.size)


# def load_grid_from_nisar_gslc(
#     gslc_path: object, subdataset: str | None, epsg: int
# ) -> FullFrameGrid:
#     """Return the authoritative native grid of a NISAR GSLC.

#     This is the grid that dolphin's ``make_nodata_mask`` and our own staged
#     GTiffs inherit. Block windows, nodata crops, and per-block bounds all
#     derive from this so ``combine_mask_files`` sees consistent raster sizes
#     across its inputs.

#     The EPSG is taken from the caller (typically ``cfg.output_options.bounds_epsg``)
#     rather than re-derived from the GSLC's projection WKT — NISAR GSLC WKTs
#     often lack an ``AUTHORITY`` tag and can defeat ``AutoIdentifyEPSG``.
#     """
#     from osgeo import gdal

#     src_str = str(gslc_path)

#     # Try to get geotransform from NISAR HDF5 metadata (handles NISAR properly)
#     gt = None
#     frequency = "frequencyA"
#     if subdataset:
#         # Extract frequency from subdataset path if present
#         if "frequency" in subdataset.lower():
#             for part in subdataset.split("/"):
#                 if part.startswith("frequency"):
#                     frequency = part
#                     break

#     if src_str.lower().endswith((".h5", ".hdf5")):
#         gt = _get_nisar_geotransform(src_str, frequency=frequency)
#         if gt is not None:
#             logger.debug(f"Got geotransform from NISAR metadata: {gt}")

#     # Fallback to standard GDAL if direct HDF5 read didn't work
#     if gt is None or tuple(gt) == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
#         logger.debug("Falling back to standard GDAL GetGeoTransform")
#         if subdataset and src_str.lower().endswith((".h5", ".hdf5", ".nc")):
#             uri = f'HDF5:"{src_str}"://{subdataset.lstrip("/")}'
#         else:
#             uri = src_str
#         ds = gdal.Open(uri)
#         if ds is None:
#             raise RuntimeError(f"GDAL could not open NISAR GSLC at {uri}")
#         try:
#             gt = ds.GetGeoTransform()
#             rows = ds.RasterYSize
#             cols = ds.RasterXSize
#         finally:
#             ds = None
#     else:
#         # Got valid geotransform from HDF5 metadata, now get dimensions via GDAL
#         if subdataset and src_str.lower().endswith((".h5", ".hdf5", ".nc")):
#             uri = f'HDF5:"{src_str}"://{subdataset.lstrip("/")}'
#         else:
#             uri = src_str
#         ds = gdal.Open(uri)
#         if ds is None:
#             raise RuntimeError(f"GDAL could not open NISAR GSLC at {uri}")
#         try:
#             rows = ds.RasterYSize
#             cols = ds.RasterXSize
#         finally:
#             ds = None

#     # Validate geotransform
#     if gt is None or tuple(gt) == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
#         raise RuntimeError(
#             f"Could not get valid geotransform from {src_str}. "
#             "Got identity matrix, which means the file is not properly georeferenced."
#         )

#     left = gt[0]
#     x_res = gt[1]
#     top = gt[3]
#     y_res = abs(gt[5])
#     right = left + cols * x_res
#     bottom = top - rows * y_res

#     logger.info(
#         f"Loaded NISAR GSLC grid: {cols}x{rows} pixels, "
#         f"bounds: ({left:.2f}, {bottom:.2f}, {right:.2f}, {top:.2f}), "
#         f"resolution: {x_res:.2f} x {y_res:.2f}"
#     )

#     return FullFrameGrid(
#         bounds=Bbox(left, bottom, right, top),
#         epsg=int(epsg),
#         x_res=x_res,
#         y_res=y_res,
#         rows=rows,
#         cols=cols,
#     )


def _narrow_cfg_for_block(
    cfg: DisplacementWorkflow,
    frame: FullFrameGrid,
    block: BlockWindow,
    block_work_dir: Path,
) -> DisplacementWorkflow:
    """Return a deep copy of ``cfg`` set up to run PL for a single block.

    The copy narrows `output_options.bounds` to the block's read window,
    disables unwrap and timeseries inversion (those run once on the assembled
    frame), and redirects PS, phase_linking, and interferogram_network outputs
    to ``block_work_dir``.

    PS, phase_linking, and interferogram_network directories are redirected to
    the block directory. Each block writes its own interferograms which are
    later assembled to the main directory. unwrap_options and timeseries_options
    remain pointing to the main work directory since those stages run on the
    assembled full frame.
    """
    block_cfg = copy.deepcopy(cfg)
    block_cfg.output_options.bounds = tuple(block_bounds(frame, block))
    block_cfg.unwrap_options.run_unwrap = False
    block_cfg.timeseries_options.run_inversion = False
    block_cfg.timeseries_options.run_velocity = False

    # Update work directory for the block
    old_work_dir = block_cfg.work_directory
    block_cfg.work_directory = block_work_dir

    # Redirect ps_options, phase_linking, and interferogram_network to block
    # directory. Each block writes its own interferograms which are later
    # assembled to the main directory.
    for step in ["ps_options", "phase_linking", "interferogram_network"]:
        opts = getattr(block_cfg, step)
        # Get the relative path from the old work directory
        try:
            rel_dir = opts._directory.relative_to(old_work_dir)
        except ValueError:
            # If not relative, just use the directory name
            rel_dir = opts._directory.name
        # Set to block work directory
        opts._directory = block_work_dir / rel_dir

    # Update PS output file paths to block directory
    ps_opts = block_cfg.ps_options
    try:
        ps_opts._output_file = block_work_dir / ps_opts._output_file.relative_to(
            old_work_dir
        )
        ps_opts._amp_mean_file = block_work_dir / ps_opts._amp_mean_file.relative_to(
            old_work_dir
        )
        ps_opts._amp_dispersion_file = (
            block_work_dir / ps_opts._amp_dispersion_file.relative_to(old_work_dir)
        )
    except ValueError:
        # Fallback if paths aren't relative
        ps_opts._output_file = block_work_dir / "PS" / ps_opts._output_file.name
        ps_opts._amp_mean_file = block_work_dir / "PS" / ps_opts._amp_mean_file.name
        ps_opts._amp_dispersion_file = (
            block_work_dir / "PS" / ps_opts._amp_dispersion_file.name
        )

    # The log file is anchored to work_directory in dolphin
    block_cfg.log_file = block_work_dir / "dolphin.log"

    return block_cfg


def _is_remote_path(p: object) -> bool:
    return str(p).startswith(_REMOTE_PATH_PREFIXES)


def build_frame_nodata_mask(
    cslc_file_list: Sequence[object],
    subdataset: str | None,
    out_file: Path,
    buffer_pixels: int = 400,
) -> Path | None:
    """Build a frame-wide nodata mask from the NISAR bounding polygons.

    Called *before* any block runs and *before* inputs are staged as GTiffs —
    the polygon metadata only lives on the original HDF5s. `opera_utils`
    reads the NISAR bounding polygon via GDAL's multidim API, which supports
    both local files and ``/vsis3/`` paths, so this works in both local and
    streaming modes.

    Returns the output path on success. Returns ``None`` if the polygon
    lookup fails (e.g. inputs are non-NISAR test files); callers should
    treat this as "no nodata mask available" and rely on the per-block
    bounds mask alone.
    """
    non_compressed = [f for f in cslc_file_list if "compressed" not in str(f).lower()]
    if not non_compressed:
        return None
    out_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        from opera_utils import make_nodata_mask

        make_nodata_mask(
            opera_file_list=non_compressed,
            out_file=out_file,
            dset_name=subdataset,
            buffer_pixels=buffer_pixels,
        )
    except Exception as e:  # noqa: BLE001 — log and fall back to bounds-only
        logger.warning("Frame nodata mask could not be built: %s", e)
        return None
    if not out_file.exists():
        return None
    return out_file


def _crop_frame_mask_to_block(
    frame_mask: Path,
    template: Path,
    block: BlockWindow,
    out_path: Path,
    frame: FullFrameGrid,
) -> Path:
    """Crop a full-frame mask to a block's rows with correct projection info.

    Extracts the block's rows from the full-frame mask and writes with a
    geotransform computed from ``frame`` to ensure the cropped mask aligns
    pixel-for-pixel with the block's bounds in the projected coordinate system.

    The output geotransform is derived from the frame grid and block window,
    ensuring consistency with the projection specified by ``frame.epsg``.

    Parameters
    ----------
    frame_mask : Path
        Full-frame mask file
    template : Path
        One of the block's staged GTiffs (used only for projection WKT)
    block : BlockWindow
        Block window defining which rows to extract
    out_path : Path
        Output path for cropped mask
    frame : FullFrameGrid
        Full frame grid with projection info and pixel spacing

    Returns
    -------
    Path
        Path to the cropped mask file

    """
    from osgeo import gdal

    arr = io.load_gdal(
        frame_mask,
        rows=slice(block.read_start, block.read_stop),
        cols=slice(None),
    )

    # Get projection WKT from template
    ds = gdal.Open(str(template))
    if ds is None:
        raise RuntimeError(f"Could not open staged template for mask crop: {template}")
    try:
        proj = ds.GetProjection()
    finally:
        ds = None

    # Compute geotransform for the block based on frame grid
    # The block starts at row block.read_start in the full frame
    gt = frame.geotransform
    block_geotransform = (
        gt[0],  # X origin (left) - same as full frame
        gt[1],  # X pixel size
        gt[2],  # X rotation (typically 0)
        gt[3] + block.read_start * gt[5],  # Y origin adjusted for block start row
        gt[4],  # Y rotation (typically 0)
        gt[5],  # Y pixel size (negative for north-up)
    )

    io.write_arr(
        arr=arr,
        output_name=out_path,
        geotransform=block_geotransform,
        projection=proj,
        dtype=arr.dtype,
        nbands=1,
    )
    return out_path


def _stage_input_to_local(
    src_path: str,
    subdataset: str | None,
    block: BlockWindow,
    out_dir: Path,
    frame: FullFrameGrid,
) -> Path:
    """Extract a block's azimuth window from ``src_path`` into a local GTiff.

    Opens the input via GDAL's HDF5 driver (if a subdataset is given) or
    directly (for plain rasters / compressed SLCs already in GTiff form),
    then uses ``gdal.Translate`` with ``srcWin`` to write only the block's
    rows to a local GeoTIFF. After extraction, the geotransform is updated
    to match the frame grid exactly, ensuring consistent georeferencing across
    all blocks.

    Parameters
    ----------
    src_path : str
        Path to source file (may be remote: /vsis3/..., s3://..., etc.)
    subdataset : str | None
        HDF5 subdataset path, if applicable
    block : BlockWindow
        Block window defining which rows to extract
    out_dir : Path
        Output directory for staged file
    frame : FullFrameGrid
        Full frame grid with projection info and pixel spacing

    Returns
    -------
    Path
        Path to the staged GTiff with correct projection information

    """
    src_str = str(src_path)
    src_name = Path(src_str.split("://")[-1]).name  # strip /vsis3/ etc.
    stem = Path(src_name).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_block{block.block_index:02d}.tif"
    if out_path.exists():
        return out_path

    # Build the GDAL open URI. HDF5 subdataset syntax is HDF5:"path":/subds.
    if subdataset and src_name.lower().endswith((".h5", ".hdf5", ".nc")):
        uri = f'HDF5:"{src_str}"://{subdataset.lstrip("/")}'
    else:
        uri = src_str

    src_ds = gdal.Open(uri)
    if src_ds is None:
        raise RuntimeError(f"GDAL could not open {uri}")
    try:
        rows = src_ds.RasterYSize
        cols = src_ds.RasterXSize
        y_off = max(0, block.read_start)
        y_stop = min(block.read_stop, rows)
        y_size = y_stop - y_off
        if y_size <= 0:
            raise ValueError(
                f"Empty window for block {block.block_index} against {src_str}: "
                f"rows={rows}, read=[{block.read_start}, {block.read_stop})"
            )
        out_ds = gdal.Translate(
            str(out_path),
            src_ds,
            format="GTiff",
            srcWin=[0, y_off, cols, y_size],
            creationOptions=[
                "TILED=YES",
                "BLOCKXSIZE=256",
                "BLOCKYSIZE=256",
                "BIGTIFF=IF_SAFER",
            ],
        )

        gt = frame.geotransform
        # Compute geotransform for this block's rows
        block_geotransform = (
            gt[0],  # X origin (left) - same as full frame
            gt[1],  # X pixel size
            gt[2],  # X rotation (typically 0)
            gt[3] + block.read_start * gt[5],  # Y origin adjusted for block start
            gt[4],  # Y rotation (typically 0)
            gt[5],  # Y pixel size (negative for north-up)
        )
        out_ds.SetGeoTransform(block_geotransform)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(frame.epsg)
        out_ds.SetProjection(srs.ExportToWkt())
    finally:
        src_ds.Close()
        # src_ds = None

    # # Update geotransform to match frame grid exactly
    # # This ensures the block's output has correct georeferencing in the
    # # projection specified by frame.epsg
    # out_ds = gdal.Open(str(out_path), gdal.GA_Update)
    # if out_ds is not None:
    #     try:
    #         gt = frame.geotransform
    #         # Compute geotransform for this block's rows
    #         block_geotransform = (
    #             gt[0],  # X origin (left) - same as full frame
    #             gt[1],  # X pixel size
    #             gt[2],  # X rotation (typically 0)
    #             gt[3] + block.read_start * gt[5],  # Y origin adjusted for block start
    #             gt[4],  # Y rotation (typically 0)
    #             gt[5],  # Y pixel size (negative for north-up)
    #         )
    #         out_ds.SetGeoTransform(block_geotransform)
    #         # Also ensure projection is set (some HDF5 sources may lack it)
    #         from osgeo import osr

    #         srs = osr.SpatialReference()
    #         srs.ImportFromEPSG(frame.epsg)
    #         out_ds.SetProjection(srs.ExportToWkt())
    #     finally:
    #         out_ds = None

    return out_path


def _stage_inputs_for_block(
    cfg: DisplacementWorkflow,
    block: BlockWindow,
    staging_dir: Path,
    frame: FullFrameGrid,
) -> tuple[list[Path], str | None]:
    """Stage every entry in ``cfg.cslc_file_list`` that points at remote storage.

    Returns (new_file_list, new_subdataset). Local paths are passed through
    unchanged. When any input is staged, the returned subdataset is ``None``
    because the staged GTiffs carry their band directly with no HDF5 subdataset
    hierarchy — the caller must set ``block_cfg.input_options.subdataset`` to
    this value.

    Parameters
    ----------
    cfg : DisplacementWorkflow
        Workflow configuration
    block : BlockWindow
        Block window defining which rows to extract
    staging_dir : Path
        Directory for staged files
    frame : FullFrameGrid
        Full frame grid with projection info and pixel spacing

    Returns
    -------
    tuple[list[Path], str | None]
        Staged file paths and new subdataset value

    """
    t0 = time.perf_counter()
    staging_dir.mkdir(parents=True, exist_ok=True)
    subdataset = cfg.input_options.subdataset
    staged: list[Path] = []
    # Always stage files (even local ones) to ensure consistent dimensions
    # across all inputs and masks. This prevents dimension mismatches when
    # dolphin creates a bounds mask using a staged file as a template.
    for src in cfg.cslc_file_list:
        staged.append(
            _stage_input_to_local(str(src), subdataset, block, staging_dir, frame)
        )
    # All files are staged as GTiffs, so no subdataset
    new_subdataset = None
    logger.info(
        "staging block completed",
        extra={
            "block": block,
            "elapsed": time.perf_counter() - t0,
        },
    )
    return staged, new_subdataset


def run_phase_linking_block(
    cfg: DisplacementWorkflow,
    frame: FullFrameGrid,
    block: BlockWindow,
    shard_dir: Path,
    debug: bool = False,
    frame_nodata_mask: Path | None = None,
    layover_shadow_mask: Path | None = None,
) -> OutputPaths:
    """Run dolphin's displacement workflow for a single azimuth block (PL only).

    IMPORTANT FOR NISAR PROCESSING:
    - NISAR has NO bursts (unlike Sentinel-1)
    - Each azimuth block is processed as a single unit
    - cfg.worker_settings.n_parallel_bursts controls parallelism WITHIN the block
      (for processing tiles/pixels in parallel), NOT between blocks
    - Block-level parallelism is controlled by _run_phase_linking_blocks()

    When inputs are remote (``/vsis3/``, ``s3://``, etc.), each GSLC and
    compressed-SLC file has the block's azimuth window materialized into a
    local GeoTIFF before dolphin starts. The staged files are deleted after
    the block completes so only one block's worth of staged data is on disk
    at a time.

    ``frame_nodata_mask`` is an optional pre-built full-frame mask produced
    by :func:`build_frame_nodata_mask`. When provided, it is cropped to this
    block's azimuth window (aligned to the staged GTiff grid) and handed to
    dolphin via ``layover_shadow_mask_files``. This preserves the NISAR
    polygon-based nodata mask across the block split, since staged GTiffs
    otherwise carry no bounding-polygon metadata.
    """
    block_work_dir = shard_dir / f"block_{block.block_index:02d}"
    block_work_dir.mkdir(parents=True, exist_ok=True)
    block_cfg = _narrow_cfg_for_block(cfg, frame, block, block_work_dir)

    staging_dir = block_work_dir / "staged_inputs"
    staged_files, new_subdataset = _stage_inputs_for_block(
        cfg, block, staging_dir, frame
    )
    if staged_files != list(cfg.cslc_file_list):
        logger.info(
            "Staged %d/%d inputs to %s for block %d",
            sum(1 for a, b in zip(cfg.cslc_file_list, staged_files) if a != b),
            len(cfg.cslc_file_list),
            staging_dir,
            block.block_index,
        )
    block_cfg.cslc_file_list = staged_files
    block_cfg.input_options.subdataset = new_subdataset

    # Crop and combine masks for this block
    block_masks = []
    if frame_nodata_mask is not None:
        template = next(
            (p for p in staged_files if _stem_looks_like_nisar(p)),
            staged_files[0],
        )
        block_nodata_mask = block_work_dir / "nodata_mask.tif"
        _crop_frame_mask_to_block(
            frame_nodata_mask, template, block, block_nodata_mask, frame
        )
        block_masks.append(block_nodata_mask)
        logger.info(
            "Cropped frame nodata mask to block %d window -> %s",
            block.block_index,
            block_nodata_mask,
        )

    if layover_shadow_mask is not None:
        template = next(
            (p for p in staged_files if _stem_looks_like_nisar(p)),
            staged_files[0],
        )
        block_layover_shadow = block_work_dir / "layover_shadow_mask.tif"
        _crop_frame_mask_to_block(
            layover_shadow_mask, template, block, block_layover_shadow, frame
        )
        block_masks.append(block_layover_shadow)
        logger.info(
            "Cropped layover/shadow mask to block %d window -> %s",
            block.block_index,
            block_layover_shadow,
        )

    # Assign combined masks to block config
    if block_masks:
        block_cfg.layover_shadow_mask_files = block_masks

    logger.info(
        "Running phase linking for block %d (rows %d..%d write, %d..%d read)",
        block.block_index,
        block.write_start,
        block.write_stop,
        block.read_start,
        block.read_stop,
    )
    try:
        # Run wrapped phase estimation only (no unwrapping/timeseries/stitching)
        # IMPORTANT: For NISAR, we call wrapped_phase.run() directly (NOT
        # displacement.run()). This processes a single "unit" (our azimuth block)
        # with NO burst-level parallelism. The max_workers parameter controls
        # parallelism WITHIN the block for processing tiles/pixels.
        # n_parallel_bursts is misleadingly named for NISAR - it actually means
        # "workers within block" not "parallel bursts".
        wrapped_output = wrapped_phase.run(
            cfg=block_cfg,
            debug=debug,
            raise_on_empty=False,
            max_workers=cfg.worker_settings.n_parallel_bursts,
        )

        # NISAR inputs have no burst id, so use "phase_linking" as the key
        burst_key = "phase_linking"

        # Convert to OutputPaths format expected by assemble_full_frame
        # Note: Correlation files are NOT generated at block level - they will be
        # generated after full-frame assembly to avoid redundant computation
        return OutputPaths(
            comp_slc_dict={burst_key: wrapped_output.comp_slc_file_list},
            stitched_ifg_paths=wrapped_output.ifg_file_list,
            stitched_cor_paths=[],  # Empty - correlations generated after assembly
            stitched_temp_coh_files=wrapped_output.temp_coh_files,
            stitched_shp_count_files=wrapped_output.shp_count_files,
            stitched_similarity_files=wrapped_output.similarity_files,
            stitched_crlb_files=wrapped_output.crlb_files,
            stitched_closure_phase_files=wrapped_output.closure_phase_files,
            stitched_ps_file=wrapped_output.ps_looked_file,
            stitched_amp_dispersion_file=wrapped_output.amp_disp_looked_file,
            unwrapped_paths=None,
            conncomp_paths=None,
            timeseries_paths=None,
            timeseries_residual_paths=None,
            reference_point=None,
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _stem_looks_like_nisar(p: object) -> bool:
    return "NISAR" in Path(str(p)).stem.upper()


def _free_block_memory() -> None:
    """Release caches/memory held after a block finishes.

    Important for the sequential in-process path on AWS Batch where one job
    runs all blocks back-to-back: without this, GDAL's /vsis3/ range-cache and
    block cache grow across blocks and can exhaust instance memory. Also drops
    any Python-level references cycled up during phase linking.
    """
    gc.collect()
    try:
        from osgeo import gdal

        # Per-file VSI curl cache and the global block cache.
        gdal.VSICurlClearCache()
        # Emptying the block cache forces GDAL to drop decompressed chunks.
        gdal.SetCacheMax(0)
        gdal.SetCacheMax(4 * 1024 * 1024 * 1024)  # restore 4 GiB default
    except ImportError:
        # GDAL not available (should never happen at runtime, but keep the
        # helper testable without GDAL in some environments).
        pass


def _run_phase_linking_blocks(
    cfg: DisplacementWorkflow,
    frame: FullFrameGrid,
    blocks: Sequence[BlockWindow],
    shard_dir: Path,
    n_parallel: int,
    debug: bool,
    frame_nodata_mask: Path | None = None,
    layover_shadow_mask: Path | None = None,
) -> list[OutputPaths]:
    """Run PL for each block.

    With ``n_parallel == 1`` (the default and the orca-batch path) blocks run
    strictly sequentially and ``_free_block_memory()`` is called between
    blocks so one Batch instance can process a whole frame without caches
    growing unbounded. With ``n_parallel > 1`` a ``ProcessPoolExecutor`` is
    used and cleanup happens naturally when each worker process exits.
    """
    n_parallel = max(1, min(n_parallel, len(blocks)))

    if n_parallel == 1:
        outputs: list[OutputPaths] = []
        for i, block in enumerate(blocks):
            outputs.append(
                run_phase_linking_block(
                    cfg,
                    frame,
                    block,
                    shard_dir,
                    debug=debug,
                    frame_nodata_mask=frame_nodata_mask,
                    layover_shadow_mask=layover_shadow_mask,
                )
            )
            if i < len(blocks) - 1:
                logger.info(
                    "Releasing memory before block %d of %d", i + 1, len(blocks)
                )
                _free_block_memory()
        return outputs

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_parallel, mp_context=ctx) as exc:
        futures = [
            exc.submit(
                run_phase_linking_block,
                cfg,
                frame,
                block,
                shard_dir,
                debug,
                frame_nodata_mask,
                layover_shadow_mask,
            )
            for block in blocks
        ]
        return [f.result() for f in futures]


@dataclass
class StitchedFramePaths:
    """Full-frame rasters produced by `stitch_full_frame`.

    Field names match `dolphin.workflows.displacement.OutputPaths` where they
    overlap, so the downstream unwrap/timeseries/products code can consume
    either object with the same accessors.
    """

    stitched_ifg_paths: list[Path]
    stitched_cor_paths: list[Path]
    stitched_temp_coh_files: list[Path]
    stitched_ps_file: Path
    stitched_amp_dispersion_file: Path
    stitched_shp_count_files: list[Path]
    stitched_similarity_files: list[Path]
    comp_slc_dict: dict[str, list[Path]]

    @property
    def stitched_temp_coh_file(self) -> Path:
        return self.stitched_temp_coh_files[-1]

    @property
    def stitched_shp_count_file(self) -> Path:
        return self.stitched_shp_count_files[-1]

    @property
    def stitched_similarity_file(self) -> Path:
        return self.stitched_similarity_files[-1]


def stitch_full_frame(
    block_outputs: Sequence[OutputPaths],
    blocks: Sequence[BlockWindow],
    out_dir: Path,
) -> StitchedFramePaths:
    """Stitch azimuth blocks together using dolphin's spatial stitching.

    Treats each azimuth block as a spatial subset (like a burst) and uses
    dolphin's stitching functions to merge them into full-frame VRT outputs.
    This is faster and more efficient than copying data into GeoTIFFs.

    Every `OutputPaths.stitched_*` field is assembled. The per-block lists
    (ifgs, correlations, temp-coh, SHP counts, similarity) must match in length
    and ordering across blocks — they correspond to the same interferogram
    network / ministacks, just evaluated over different azimuth windows.
    """
    from dolphin import stitching
    from dolphin.io import EXTRA_COMPRESSED_TIFF_OPTIONS  # DEFAULT_TIFF_OPTIONS

    DEFAULT_TIFF_OPTIONS_RIO = [
        "COMPRESS=ZSTD",
        "ZSTD_LEVEL=1",  # fast, still ~LZW-level
        "PREDICTOR=2",  # or 3 for floats; skip for complex types
        "TILED=YES",
        "BLOCKXSIZE=256",
        "BLOCKYSIZE=256",
        "BIGTIFF=IF_SAFER",
        "NUM_THREADS=ALL_CPUS",
    ]
    DEFAULT_TIFF_OPTIONS = tuple(
        f"{k.upper()}={v}" for k, v in DEFAULT_TIFF_OPTIONS_RIO.items()
    )

    if len(block_outputs) != len(blocks):
        raise ValueError(
            f"Got {len(block_outputs)} block outputs but {len(blocks)} block windows"
        )

    # Collect all interferograms from all blocks (flatten the lists)
    all_ifg_files = []
    for block_output in block_outputs:
        all_ifg_files.extend(block_output.stitched_ifg_paths)

    # Stitch interferograms by date using dolphin's burst stitching approach
    # Use GeoTIFF format for self-contained, portable outputs that allow
    # block files to be cleaned up after stitching
    # Use DEFAULT_TIFF_OPTIONS for complex data (interferograms are CFloat32)
    logger.info("Stitching interferograms from %d azimuth blocks", len(blocks))
    ifg_dir = out_dir / "interferograms"
    ifg_dir.mkdir(parents=True, exist_ok=True)
    date_to_ifg = stitching.merge_by_date(
        image_file_list=all_ifg_files,
        file_date_fmt="%Y%m%d",
        output_dir=ifg_dir,
        output_suffix=".int.tif",
        num_workers=3,
        options=DEFAULT_TIFF_OPTIONS,
    )
    stitched_ifg_paths = list(date_to_ifg.values())

    # Generate interferometric correlations from stitched interferograms
    logger.info("Generating interferometric correlations for stitched frame")
    corr_window_size = (11, 11)
    stitched_cor_paths = interferogram.estimate_interferometric_correlations(
        ifg_filenames=stitched_ifg_paths,
        window_size=corr_window_size,
        num_workers=3,
        options=EXTRA_COMPRESSED_TIFF_OPTIONS,
    )

    # Stitch temporal coherence files
    all_temp_coh = []
    for block_output in block_outputs:
        all_temp_coh.extend(block_output.stitched_temp_coh_files)

    logger.info("Stitching temporal coherence files")
    date_to_temp_coh = stitching.merge_by_date(
        image_file_list=all_temp_coh,
        file_date_fmt="%Y%m%d",
        output_dir=ifg_dir,
        output_prefix="auto",
        num_workers=3,
        options=EXTRA_COMPRESSED_TIFF_OPTIONS,
    )
    stitched_temp_coh_files = list(date_to_temp_coh.values())

    # Stitch SHP count files
    all_shp_counts = []
    for block_output in block_outputs:
        all_shp_counts.extend(block_output.stitched_shp_count_files)

    logger.info("Stitching SHP count files")
    date_to_shp_count = stitching.merge_by_date(
        image_file_list=all_shp_counts,
        file_date_fmt="%Y%m%d",
        output_dir=ifg_dir,
        output_prefix="auto",
        num_workers=3,
    )
    stitched_shp_count_files = list(date_to_shp_count.values())

    # Stitch similarity files
    all_similarity = []
    for block_output in block_outputs:
        all_similarity.extend(block_output.stitched_similarity_files)

    logger.info("Stitching similarity files")
    date_to_similarity = stitching.merge_by_date(
        image_file_list=all_similarity,
        file_date_fmt="%Y%m%d",
        output_dir=ifg_dir,
        output_prefix="auto",
        resample_alg="nearest",
        num_workers=3,
    )
    stitched_similarity_files = list(date_to_similarity.values())

    # Stitch PS mask files (single file, not by date)
    ps_file_list = [block_output.stitched_ps_file for block_output in block_outputs]
    stitched_ps_file = ifg_dir / "ps_mask_looked.tif"
    logger.info("Stitching PS mask files")
    if not stitched_ps_file.exists():
        stitching.merge_images(
            ps_file_list,
            outfile=stitched_ps_file,
            out_nodata=255,
            resample_alg="nearest",
        )

    # Stitch amplitude dispersion files (single file, not by date)
    amp_disp_list = [
        block_output.stitched_amp_dispersion_file for block_output in block_outputs
    ]
    stitched_amp_dispersion_file = ifg_dir / "amp_dispersion_looked.tif"
    logger.info("Stitching amplitude dispersion files")
    if not stitched_amp_dispersion_file.exists():
        stitching.merge_images(
            amp_disp_list,
            outfile=stitched_amp_dispersion_file,
            resample_alg="nearest",
        )

    # Stitch compressed SLCs
    comp_slc_dict: dict[str, list[Path]] = {}
    burst_keys = list(block_outputs[0].comp_slc_dict.keys())
    for burst in burst_keys:
        # Collect all compressed SLCs for this burst across all blocks
        all_comp_slcs = []
        for block_output in block_outputs:
            all_comp_slcs.extend(block_output.comp_slc_dict[burst])

        # Stitch by date
        # Use DEFAULT_TIFF_OPTIONS for complex data (not EXTRA_COMPRESSED which has
        # NBITS/PREDICTOR that don't work with CFloat32)
        logger.info(f"Stitching compressed SLCs for burst {burst}")
        comp_slc_dir = ifg_dir / "compressed_slcs"
        comp_slc_dir.mkdir(exist_ok=True, parents=True)
        date_to_comp_slc = stitching.merge_by_date(
            image_file_list=all_comp_slcs,
            file_date_fmt="%Y%m%d",
            output_dir=comp_slc_dir,
            output_suffix=".tif",
            num_workers=3,
            options=DEFAULT_TIFF_OPTIONS,
        )
        comp_slc_dict[burst] = list(date_to_comp_slc.values())

    logger.info("Finished stitching all azimuth blocks into full-frame outputs")
    return StitchedFramePaths(
        stitched_ifg_paths=stitched_ifg_paths,
        stitched_cor_paths=stitched_cor_paths,
        stitched_temp_coh_files=stitched_temp_coh_files,
        stitched_ps_file=stitched_ps_file,
        stitched_amp_dispersion_file=stitched_amp_dispersion_file,
        stitched_shp_count_files=stitched_shp_count_files,
        stitched_similarity_files=stitched_similarity_files,
        comp_slc_dict=comp_slc_dict,
    )


def run_full_frame_unwrap_and_timeseries(
    cfg: DisplacementWorkflow, stitched: StitchedFramePaths
) -> OutputPaths:
    """Unwrap the stitched ifgs and invert the timeseries on the full frame.

    Uses the interferometric correlations generated during stitching.
    Mirrors the last two stages of `dolphin.workflows.displacement.run` so the
    returned `OutputPaths` drops into disp-nisar's existing `create_products`.
    """
    from dolphin import timeseries
    from dolphin.workflows import unwrapping

    avg_temp_coh_file = stitched.stitched_temp_coh_files[-1]
    full_similarity_file = stitched.stitched_similarity_files[-1]

    row_looks, col_looks = cfg.phase_linking.half_window.to_looks()
    nlooks = row_looks * col_looks

    unwrapped_paths, conncomp_paths = unwrapping.run(
        ifg_file_list=stitched.stitched_ifg_paths,
        cor_file_list=stitched.stitched_cor_paths,
        temporal_coherence_filename=avg_temp_coh_file,
        similarity_filename=full_similarity_file,
        nlooks=nlooks,
        unwrap_options=cfg.unwrap_options,
        mask_file=cfg.mask_file,
    )

    ts_opts = cfg.timeseries_options
    if ts_opts.run_inversion or ts_opts.run_velocity:
        timeseries_paths, timeseries_residual_paths, reference_point = timeseries.run(
            unwrapped_paths=unwrapped_paths,
            conncomp_paths=conncomp_paths,
            corr_paths=stitched.stitched_cor_paths,
            reference_point=cfg.timeseries_options.reference_point,
            quality_file=avg_temp_coh_file,
            reference_candidate_threshold=0.95,
            output_dir=ts_opts._directory,
            method=timeseries.InversionMethod(ts_opts.method),
            run_velocity=ts_opts.run_velocity,
            velocity_file=ts_opts._velocity_file,
            mask_path=cfg.mask_file if ts_opts.apply_mask_to_timeseries else None,
            correlation_threshold=ts_opts.correlation_threshold,
            num_threads=ts_opts.num_parallel_blocks,
            wavelength=cfg.input_options.wavelength,
            add_overviews=cfg.output_options.add_overviews,
            extra_reference_date=cfg.output_options.extra_reference_date,
        )
    else:
        timeseries_paths = None
        timeseries_residual_paths = None
        reference_point = None

    return OutputPaths(
        comp_slc_dict=stitched.comp_slc_dict,
        stitched_ifg_paths=stitched.stitched_ifg_paths,
        stitched_cor_paths=stitched.stitched_cor_paths,
        stitched_temp_coh_files=stitched.stitched_temp_coh_files,
        stitched_shp_count_files=stitched.stitched_shp_count_files,
        stitched_similarity_files=stitched.stitched_similarity_files,
        stitched_crlb_files=[],
        stitched_closure_phase_files=[],
        stitched_ps_file=stitched.stitched_ps_file,
        stitched_amp_dispersion_file=stitched.stitched_amp_dispersion_file,
        unwrapped_paths=unwrapped_paths,
        conncomp_paths=conncomp_paths,
        timeseries_paths=timeseries_paths,
        timeseries_residual_paths=timeseries_residual_paths,
        reference_point=reference_point,
    )


def load_block_outputs_from_shards(
    shard_dir: Path, num_blocks: int
) -> list[OutputPaths]:
    """Reconstruct per-block `OutputPaths` from a populated ``shard_dir``.

    Used by the batch finalize step, where each block was produced by a separate
    Batch job and written under ``shard_dir/block_k/``. We locate the
    corresponding files by re-running dolphin's directory layout conventions.
    """
    block_outputs: list[OutputPaths] = []
    for k in range(num_blocks):
        block_dir = shard_dir / f"block_{k:02d}"
        if not block_dir.is_dir():
            raise FileNotFoundError(f"Missing block shard directory: {block_dir}")
        block_outputs.append(_collect_block_output_paths(block_dir))
    return block_outputs


def _collect_block_output_paths(block_dir: Path) -> OutputPaths:
    """Collect the scratched OutputPaths layout produced by `run_displacement`.

    Dolphin's scratch layout under the workflow's ``work_directory`` is:

    * ``interferograms/``          — stitched top-level outputs
      (``*.int.tif``, ``*.cor.tif``, ``temporal_coherence*.tif``,
       ``shp_counts*.tif``, ``similarity*.tif``, ``ps_mask_looked.tif``,
       ``amp_dispersion_looked.tif``)
    * ``<burst_id>/linked_phase/`` — per-burst compressed SLCs
      (``compressed_*.tif``). For NISAR the single burst id is
      ``phase_linking`` (see ``dolphin.workflows.displacement.run``
      fallback in displacement.py:97).
    """

    def _sorted(pattern: str) -> list[Path]:
        return sorted(block_dir.glob(pattern))

    ifg_paths = _sorted("interferograms/*.int.tif")
    cor_paths = _sorted("interferograms/*.cor.tif")
    temp_coh = _sorted("interferograms/temporal_coherence*.tif")
    shp_count = _sorted("interferograms/shp_counts*.tif")
    similarity = _sorted("interferograms/similarity*.tif")
    ps_file = next(block_dir.glob("interferograms/ps_mask_looked.tif"), None)
    amp_disp = next(block_dir.glob("interferograms/amp_dispersion_looked.tif"), None)
    # Compressed SLCs are one level down in a burst-named subdirectory.
    comp_slcs = sorted(block_dir.glob("*/linked_phase/compressed_*.tif"))

    missing = [
        name
        for name, val in (
            ("ifg", ifg_paths),
            ("cor", cor_paths),
            ("temp_coh", temp_coh),
            ("ps_file", ps_file),
            ("amp_dispersion", amp_disp),
        )
        if not val
    ]
    if missing:
        raise FileNotFoundError(
            f"Block directory {block_dir} is missing expected outputs: {missing}"
        )

    # Group compressed SLCs by burst directory name.
    comp_slc_dict: dict[str, list[Path]] = {}
    for p in comp_slcs:
        # <burst_id>/linked_phase/compressed_...tif
        burst = p.parent.parent.name
        comp_slc_dict.setdefault(burst, []).append(p)

    return OutputPaths(
        comp_slc_dict=comp_slc_dict,
        stitched_ifg_paths=ifg_paths,
        stitched_cor_paths=cor_paths,
        stitched_temp_coh_files=temp_coh,
        stitched_shp_count_files=shp_count,
        stitched_similarity_files=similarity,
        stitched_crlb_files=[],
        stitched_closure_phase_files=[],
        stitched_ps_file=ps_file,  # type: ignore[arg-type]
        stitched_amp_dispersion_file=amp_disp,  # type: ignore[arg-type]
        unwrapped_paths=None,
        conncomp_paths=None,
        timeseries_paths=None,
        timeseries_residual_paths=None,
        reference_point=None,
    )


def get_nisar_pixel_spacing(gslc_file: Path, frequency: str) -> tuple[float, float]:
    """Return (x_res, y_res) in meters for a NISAR GSLC, reading HDF5 metadata."""
    import h5py

    with h5py.File(gslc_file, "r") as h5f:
        grid = h5f[f"science/LSAR/GSLC/grids/{frequency}"]
        x_spacing = float(grid["xCoordinateSpacing"][()])
        y_spacing = float(grid["yCoordinateSpacing"][()])
    return abs(x_spacing), abs(y_spacing)
