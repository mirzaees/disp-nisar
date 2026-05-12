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
import gc
import logging
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

from dolphin import interferogram, io
from dolphin._types import Bbox
from dolphin.workflows import wrapped_phase
from dolphin.workflows.config import DisplacementWorkflow
from dolphin.workflows.displacement import OutputPaths

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

    index: int
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
                index=k,
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


def load_grid_from_nisar_gslc(
    gslc_path: object, subdataset: str | None, epsg: int
) -> FullFrameGrid:
    """Return the authoritative native grid of a NISAR GSLC.

    This is the grid that dolphin's ``make_nodata_mask`` and our own staged
    GTiffs inherit. Block windows, nodata crops, and per-block bounds all
    derive from this so ``combine_mask_files`` sees consistent raster sizes
    across its inputs.

    The EPSG is taken from the caller (typically ``cfg.output_options.bounds_epsg``)
    rather than re-derived from the GSLC's projection WKT — NISAR GSLC WKTs
    often lack an ``AUTHORITY`` tag and can defeat ``AutoIdentifyEPSG``.
    """
    from osgeo import gdal

    src_str = str(gslc_path)
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
    left = gt[0]
    x_res = gt[1]
    top = gt[3]
    y_res = abs(gt[5])
    right = left + cols * x_res
    bottom = top - rows * y_res
    return FullFrameGrid(
        bounds=Bbox(left, bottom, right, top),
        epsg=int(epsg),
        x_res=x_res,
        y_res=y_res,
        rows=rows,
        cols=cols,
    )


def _narrow_cfg_for_block(
    cfg: DisplacementWorkflow,
    frame: FullFrameGrid,
    block: BlockWindow,
    block_work_dir: Path,
) -> DisplacementWorkflow:
    """Return a deep copy of ``cfg`` set up to run PL for a single block.

    The copy narrows `output_options.bounds` to the block's read window, disables
    unwrap and timeseries inversion (those run once on the assembled frame), and
    redirects all scratch to ``block_work_dir``.
    """
    block_cfg = copy.deepcopy(cfg)
    block_cfg.output_options.bounds = tuple(block_bounds(frame, block))
    block_cfg.unwrap_options.run_unwrap = False
    block_cfg.timeseries_options.run_inversion = False
    block_cfg.timeseries_options.run_velocity = False
    block_cfg.work_directory = block_work_dir
    # The log file is anchored to work_directory in dolphin — force it so the
    # per-block logs end up under the block dir.
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
    from osgeo import gdal

    src_str = str(src_path)
    src_name = Path(src_str.split("://")[-1]).name  # strip /vsis3/ etc.
    stem = Path(src_name).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_block{block.index:02d}.tif"
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
                f"Empty window for block {block.index} against {src_str}: "
                f"rows={rows}, read=[{block.read_start}, {block.read_stop})"
            )
        gdal.Translate(
            str(out_path),
            src_ds,
            format="GTiff",
            srcWin=[0, y_off, cols, y_size],
            creationOptions=[
                "COMPRESS=LZW",
                "TILED=YES",
                "BLOCKXSIZE=256",
                "BLOCKYSIZE=256",
                "BIGTIFF=IF_SAFER",
            ],
        )
    finally:
        src_ds = None

    # Update geotransform to match frame grid exactly
    # This ensures the block's output has correct georeferencing in the
    # projection specified by frame.epsg
    out_ds = gdal.Open(str(out_path), gdal.GA_Update)
    if out_ds is not None:
        try:
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
            # Also ensure projection is set (some HDF5 sources may lack it)
            from osgeo import osr

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(frame.epsg)
            out_ds.SetProjection(srs.ExportToWkt())
        finally:
            out_ds = None

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
    staging_dir.mkdir(parents=True, exist_ok=True)
    subdataset = cfg.input_options.subdataset
    staged: list[Path] = []
    # any_staged = False
    for src in cfg.cslc_file_list:
        staged.append(
            _stage_input_to_local(str(src), subdataset, block, staging_dir, frame)
        )
        # if _is_remote_path(src):
        #     any_staged = True
        #     staged.append(
        #         _stage_input_to_local(str(src), subdataset, block, staging_dir, frame)
        #     )
        # else:
        #     staged.append(Path(src))
    new_subdataset = None  # if any_staged else subdataset
    return staged, new_subdataset


def run_phase_linking_block(
    cfg: DisplacementWorkflow,
    frame: FullFrameGrid,
    block: BlockWindow,
    shard_dir: Path,
    debug: bool = False,
    frame_nodata_mask: Path | None = None,
) -> OutputPaths:
    """Run dolphin's displacement workflow for a single azimuth block (PL only).

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
    block_work_dir = shard_dir / f"block_{block.index:02d}"
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
            block.index,
        )
    block_cfg.cslc_file_list = staged_files
    block_cfg.input_options.subdataset = new_subdataset

    if frame_nodata_mask is not None:
        template = next(
            (p for p in staged_files if _stem_looks_like_nisar(p)),
            staged_files[0],
        )
        block_mask = block_work_dir / "phase_linking/nodata_mask.tif"
        block_mask.parent.mkdir(parents=True, exist_ok=True)
        _crop_frame_mask_to_block(frame_nodata_mask, template, block, block_mask, frame)
        block_cfg.layover_shadow_mask_files = [block_mask]
        logger.info(
            "Cropped frame nodata mask to block %d window -> %s",
            block.index,
            block_mask,
        )

    logger.info(
        "Running phase linking for block %d (rows %d..%d write, %d..%d read)",
        block.index,
        block.write_start,
        block.write_stop,
        block.read_start,
        block.read_stop,
    )
    try:
        # Run wrapped phase estimation only (no unwrapping/timeseries/stitching)
        wrapped_output = wrapped_phase.run(cfg=block_cfg, debug=debug)

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
            )
            for block in blocks
        ]
        return [f.result() for f in futures]


def _allocate_like(
    template: Path,
    out_path: Path,
    frame: FullFrameGrid,
    nbands: int | None = None,
) -> None:
    """Create an empty full-frame raster with `template`'s dtype/nodata and the
    full-frame geotransform/projection.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io.write_arr(
        arr=None,
        output_name=out_path,
        like_filename=template,
        shape=(frame.rows, frame.cols),
        geotransform=frame.geotransform,
        projection=frame.epsg,
        nbands=nbands,
    )


def _copy_central_rows(
    src: Path, dst: Path, block: BlockWindow, frame: FullFrameGrid
) -> None:
    """Copy the central rows of a per-block raster into the full-frame raster.

    The source raster's row-0 corresponds to the block's ``read_start`` in the
    full frame; the destination's row-0 corresponds to frame-row-0. We read the
    central-row slice from the source and write it at ``block.write_start`` in
    the destination.
    """
    src_row_start = block.write_start - block.read_start
    src_row_stop = src_row_start + block.write_height
    # Inputs from dolphin are single-band GeoTIFFs at the block's bounds.
    arr = io.load_gdal(
        src, rows=slice(src_row_start, src_row_stop), cols=slice(0, frame.cols)
    )
    if arr.ndim == 3:
        # write_block handles (bands, rows, cols) directly.
        io.write_block(arr, dst, row_start=block.write_start, col_start=0)
    else:
        io.write_block(arr, dst, row_start=block.write_start, col_start=0)


@dataclass
class AssembledFramePaths:
    """Full-frame rasters produced by `assemble_full_frame`.

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


def assemble_full_frame(
    block_outputs: Sequence[OutputPaths],
    blocks: Sequence[BlockWindow],
    frame: FullFrameGrid,
    out_dir: Path,
) -> AssembledFramePaths:
    """Pre-allocate full-frame rasters and copy each block's central rows in.

    Every `OutputPaths.stitched_*` field is assembled. The per-block lists
    (ifgs, correlations, temp-coh, SHP counts, similarity) must match in length
    and ordering across blocks — they correspond to the same interferogram
    network / ministacks, just evaluated over different azimuth windows.
    """
    if len(block_outputs) != len(blocks):
        raise ValueError(
            f"Got {len(block_outputs)} block outputs but {len(blocks)} block windows"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    def _assemble_list(attr: str) -> list[Path]:
        per_block_lists = [getattr(b, attr) for b in block_outputs]
        n = len(per_block_lists[0])
        if any(len(lst) != n for lst in per_block_lists):
            raise ValueError(
                f"Per-block lists disagree in length for {attr}:"
                f" {[len(lst) for lst in per_block_lists]}"
            )
        out_paths: list[Path] = []
        for i in range(n):
            template = Path(per_block_lists[0][i])
            out_path = out_dir / template.name
            _allocate_like(template, out_path, frame)
            for block, per_block in zip(blocks, per_block_lists):
                _copy_central_rows(Path(per_block[i]), out_path, block, frame)
            out_paths.append(out_path)
        return out_paths

    def _assemble_single(attr: str) -> Path:
        per_block = [Path(getattr(b, attr)) for b in block_outputs]
        out_path = out_dir / per_block[0].name
        _allocate_like(per_block[0], out_path, frame)
        for block, src in zip(blocks, per_block):
            _copy_central_rows(src, out_path, block, frame)
        return out_path

    stitched_ifg_paths = _assemble_list("stitched_ifg_paths")

    # Generate interferometric correlations from assembled full-frame interferograms
    # This is done here (not at block level) to avoid redundant computation on
    # overlapping halo regions between blocks
    logger.info("Generating interferometric correlations for assembled frame")
    corr_window_size = (11, 11)  # Same default as in displacement workflow
    stitched_cor_paths = interferogram.estimate_interferometric_correlations(
        ifg_filenames=stitched_ifg_paths,
        window_size=corr_window_size,
        num_workers=3,
    )

    stitched_temp_coh_files = _assemble_list("stitched_temp_coh_files")
    stitched_shp_count_files = _assemble_list("stitched_shp_count_files")
    stitched_similarity_files = _assemble_list("stitched_similarity_files")
    stitched_ps_file = _assemble_single("stitched_ps_file")
    stitched_amp_dispersion_file = _assemble_single("stitched_amp_dispersion_file")

    # Compressed SLCs are produced per-ministack with the block's narrowed bounds.
    # Assemble each ministack's compressed SLC into a full-frame version.
    comp_slc_dict: dict[str, list[Path]] = {}
    burst_keys = list(block_outputs[0].comp_slc_dict.keys())
    for burst in burst_keys:
        ministacks = [b.comp_slc_dict[burst] for b in block_outputs]
        n = len(ministacks[0])
        if any(len(lst) != n for lst in ministacks):
            raise ValueError(
                f"Compressed SLC list lengths differ across blocks for burst {burst}"
            )
        assembled: list[Path] = []
        for i in range(n):
            template = Path(ministacks[0][i])
            out_path = out_dir / "compressed_slcs" / template.name
            _allocate_like(template, out_path, frame)
            for block, per_block in zip(blocks, ministacks):
                _copy_central_rows(Path(per_block[i]), out_path, block, frame)
            assembled.append(out_path)
        comp_slc_dict[burst] = assembled

    return AssembledFramePaths(
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
    cfg: DisplacementWorkflow, assembled: AssembledFramePaths
) -> OutputPaths:
    """Unwrap the assembled ifgs and invert the timeseries on the full frame.

    Uses the interferometric correlations generated during assembly.
    Mirrors the last two stages of `dolphin.workflows.displacement.run` so the
    returned `OutputPaths` drops into disp-nisar's existing `create_products`.
    """
    from dolphin import timeseries
    from dolphin.workflows import unwrapping

    avg_temp_coh_file = assembled.stitched_temp_coh_files[-1]
    full_similarity_file = assembled.stitched_similarity_files[-1]

    row_looks, col_looks = cfg.phase_linking.half_window.to_looks()
    nlooks = row_looks * col_looks

    unwrapped_paths, conncomp_paths = unwrapping.run(
        ifg_file_list=assembled.stitched_ifg_paths,
        cor_file_list=assembled.stitched_cor_paths,
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
            corr_paths=assembled.stitched_cor_paths,
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
        comp_slc_dict=assembled.comp_slc_dict,
        stitched_ifg_paths=assembled.stitched_ifg_paths,
        stitched_cor_paths=assembled.stitched_cor_paths,
        stitched_temp_coh_files=assembled.stitched_temp_coh_files,
        stitched_shp_count_files=assembled.stitched_shp_count_files,
        stitched_similarity_files=assembled.stitched_similarity_files,
        stitched_crlb_files=[],
        stitched_closure_phase_files=[],
        stitched_ps_file=assembled.stitched_ps_file,
        stitched_amp_dispersion_file=assembled.stitched_amp_dispersion_file,
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
