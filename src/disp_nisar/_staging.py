"""Stage NISAR GSLC inputs once into compact local HDF5 files.

dolphin now performs azimuth-block splitting internally (via
``cfg.input_options.azimuth_blocks``), but its GDAL HDF5 reader needs *local*
files and the full-resolution NISAR GSLCs are large and remote. This module
repacks each input GSLC exactly once into a compact local NISAR-style HDF5 that
carries only the layers disp-nisar and dolphin actually read:

* the requested frequency/polarization raster (the ``subdataset``),
* the per-frequency grid geo datasets + ``centerFrequency``,
* the ``identification``, ``metadata/orbit`` and ``metadata/sourceData`` groups.

The repack reuses opera-utils' streaming subset (``process_file`` /
``open_h5``), which authenticates against Earthdata/S3 and streams the remote
HDF5 without a full-file download. The staged files are self-sufficient, so no
separate orbit/metadata cache is needed downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from dolphin._types import Filename

logger = logging.getLogger(__name__)

# NISAR polarization names (mirrors opera_utils.nisar._product.NISAR_POLARIZATIONS).
# Kept local so this module imports without the remote-streaming extras
# (aiohttp/fsspec/s3fs), which are only needed when staging actually runs.
_NISAR_POLARIZATIONS = ("HH", "VV", "HV", "VH", "RH", "RV", "LH", "LV")

__all__ = [
    "stage_inputs_to_local",
    "parse_frequency_and_polarization",
    "build_frame_nodata_mask",
]


def parse_frequency_and_polarization(
    subdataset: str | None,
) -> tuple[str, str]:
    """Parse the frequency letter and polarization from a NISAR subdataset path.

    ``/science/LSAR/GSLC/grids/frequencyA/HH`` -> ``("A", "HH")``. Falls back to
    ``("A", "HH")`` when the subdataset is missing or cannot be parsed.
    """
    frequency, polarization = "A", "HH"
    if not subdataset:
        return frequency, polarization
    parts = [p for p in str(subdataset).split("/") if p]
    for p in parts:
        if p.startswith("frequency") and len(p) > len("frequency"):
            frequency = p[len("frequency") :]
    if parts and parts[-1] in _NISAR_POLARIZATIONS:
        polarization = parts[-1]
    return frequency, polarization


def _is_compressed(path: object) -> bool:
    return "compressed" in str(path).lower()


def _is_nisar_hdf5(path: object) -> bool:
    return str(path).lower().endswith((".h5", ".hdf5"))


def stage_inputs_to_local(
    cslc_file_list: Sequence[Filename],
    subdataset: str | None,
    out_dir: Path,
) -> list[Path]:
    """Repack each remote/local GSLC once into a compact local HDF5.

    Parameters
    ----------
    cslc_file_list : Sequence[Filename]
        Input GSLC paths/URLs (local, ``s3://``, ``/vsis3/``, ``https://``).
    subdataset : str | None
        NISAR subdataset path (e.g. ``/science/LSAR/GSLC/grids/frequencyA/HH``).
        Determines which frequency/polarization layer is extracted.
    out_dir : Path
        Directory for the staged compact HDF5 files.

    Returns
    -------
    list[Path]
        Staged file list, in the same order as ``cslc_file_list``. Compressed
        SLCs (GTiffs from prior ministacks) and non-HDF5 entries are passed
        through unchanged.

    """
    # Imported lazily: pulls in the remote-streaming extras (aiohttp/fsspec/s3fs).
    from opera_utils.nisar._download import process_file

    frequency, polarization = parse_frequency_and_polarization(subdataset)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    staged: list[Path] = []
    for src in cslc_file_list:
        if _is_compressed(src) or not _is_nisar_hdf5(src):
            # Compressed SLCs are GTiffs without a NISAR group hierarchy; other
            # non-HDF5 inputs (e.g. test fixtures) are passed through untouched.
            staged.append(Path(str(src)))
            continue
        out = process_file(
            url=str(src),
            rows=None,
            cols=None,
            output_dir=out_dir,
            frequency=frequency,
            polarizations=[polarization],
        )
        staged.append(out)

    logger.info(
        "Staged %d inputs to %s (frequency=%s, polarization=%s)",
        len(staged),
        out_dir,
        frequency,
        polarization,
    )
    return staged


def build_frame_nodata_mask(
    cslc_file_list: Sequence[Filename],
    subdataset: str | None,
    out_file: Path,
    buffer_pixels: int = 400,
) -> Path | None:
    """Build a frame-wide nodata mask from the NISAR bounding polygons.

    ``opera_utils`` reads the NISAR bounding polygon via GDAL's multidim API,
    which works for both local files and ``/vsis3/`` paths. Compressed SLCs are
    skipped (they carry no bounding polygon).

    Returns the output path on success, or ``None`` if the polygon lookup fails
    (e.g. non-NISAR test files); callers should then rely on other masks.
    """
    non_compressed = [f for f in cslc_file_list if not _is_compressed(f)]
    if not non_compressed:
        return None
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        from opera_utils import make_nodata_mask

        make_nodata_mask(
            opera_file_list=non_compressed,
            out_file=out_file,
            dset_name=subdataset,
            buffer_pixels=buffer_pixels,
        )
    except Exception as e:  # noqa: BLE001 — log and fall back to other masks
        logger.warning("Frame nodata mask could not be built: %s", e)
        return None
    if not out_file.exists():
        return None
    return out_file
