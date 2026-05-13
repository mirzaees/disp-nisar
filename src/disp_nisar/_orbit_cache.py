"""Module for caching orbit metadata to support baseline computation without GSLC access."""

import logging
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from dolphin._types import Filename
from opera_utils import get_orbit_arrays, parse_filename

logger = logging.getLogger(__name__)


def _get_look_side_from_file(h5file: Filename) -> str:
    """Get the look side from a NISAR GSLC HDF5 file.

    Returns
    -------
    str
        "Left" or "Right"
    """
    with h5py.File(h5file, "r") as hf:
        # Try NISAR path first
        for path in [
            "/science/LSAR/identification/lookDirection",
            "/identification/lookDirection",
        ]:
            if path in hf:
                look_dir = (
                    hf[path][()].decode()
                    if isinstance(hf[path][()], bytes)
                    else hf[path][()]
                )
                return look_dir
        # Default to right if not found
        return "Right"


def save_orbit_metadata_for_cslcs(
    cslc_files: list[Filename],
    subdataset: str | None,
    output_dir: Path,
) -> None:
    """Save orbit metadata for all CSLC files to enable baseline computation later.

    Parameters
    ----------
    cslc_files : list[Filename]
        List of paths to CSLC/GSLC HDF5 files
    subdataset : str | None
        HDF5 subdataset path (if applicable)
    output_dir : Path
        Directory to save orbit metadata files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for cslc_file in cslc_files:
        # Skip compressed SLCs - they don't have orbit data
        if "compressed" in str(cslc_file).lower():
            continue

        try:
            # Parse filename to get date for naming the cache file
            parsed = parse_filename(cslc_file)
            start_dt = parsed.get("start_datetime")
            if isinstance(start_dt, datetime):
                date_str = start_dt.strftime("%Y%m%dT%H%M%S")
            else:
                # Fallback to using the filename
                date_str = Path(cslc_file).stem[:20]

            orbit_file = output_dir / f"orbit_{date_str}.npz"

            if orbit_file.exists():
                logger.debug(f"Orbit metadata already exists: {orbit_file}")
                continue

            # Get orbit arrays from the file
            times, positions, velocities, reference_epoch = get_orbit_arrays(cslc_file)

            # Get look side
            look_side = _get_look_side_from_file(cslc_file)

            # Save to compressed numpy format
            np.savez_compressed(
                orbit_file,
                times=times,
                positions=positions,
                velocities=velocities,
                reference_epoch_iso=reference_epoch.isoformat(),
                look_side=look_side,
                source_file=str(cslc_file),
            )

            logger.debug(f"Saved orbit metadata to {orbit_file}")

        except Exception as e:
            logger.warning(
                f"Failed to save orbit metadata for {cslc_file}: {e}. "
                "Baseline computation may fail if this file is needed."
            )


def load_orbit_data(orbit_cache_dir: Path, cslc_filename: Filename):
    """Load cached orbit data for a CSLC file.

    Parameters
    ----------
    orbit_cache_dir : Path
        Directory containing cached orbit files
    cslc_filename : Filename
        Original CSLC filename (used to find the matching cache file)

    Returns
    -------
    dict
        Dictionary with keys: times, positions, velocities, reference_epoch, look_side
        Or None if not found
    """
    try:
        # Parse filename to find the matching orbit cache file
        parsed = parse_filename(cslc_filename)
        start_dt = parsed.get("start_datetime")
        if isinstance(start_dt, datetime):
            date_str = start_dt.strftime("%Y%m%dT%H%M%S")
        else:
            date_str = Path(cslc_filename).stem[:20]

        orbit_file = orbit_cache_dir / f"orbit_{date_str}.npz"

        if not orbit_file.exists():
            # Try finding by glob pattern if exact match fails
            possible_files = list(orbit_cache_dir.glob(f"orbit_{date_str[:8]}*.npz"))
            if possible_files:
                orbit_file = possible_files[0]
            else:
                logger.error(f"Orbit cache file not found: {orbit_file}")
                return None

        # Load the orbit data
        data = np.load(orbit_file, allow_pickle=True)

        return {
            "times": data["times"],
            "positions": data["positions"],
            "velocities": data["velocities"],
            "reference_epoch": datetime.fromisoformat(str(data["reference_epoch_iso"])),
            "look_side": str(data["look_side"]),
        }

    except Exception as e:
        logger.error(f"Failed to load orbit data for {cslc_filename}: {e}")
        return None
