"""Module for caching GSLC metadata to support product creation.

This module caches:
- Orbit data (times, positions, velocities)
- Look direction
- Zero doppler times
- Orbit type and direction
- All HDF5 datasets needed for product metadata
"""

import json
import logging
from datetime import datetime
from os import fspath
from pathlib import Path
from typing import Any

import numpy as np
from dolphin._types import Filename
from opera_utils import get_orbit_arrays, get_zero_doppler_time, parse_filename
from opera_utils._cslc import _read_mdarray_value, _read_string_mdarray

try:
    from osgeo import gdal

    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False
    gdal = None
    osr = None

logger = logging.getLogger(__name__)

# NISAR datasets to cache for product metadata
# Note: We skip "/science/LSAR/GSLC/metadata/orbit" group since it has
# complex structure and orbit data is already cached in .npz files
NISAR_METADATA_PATHS = [
    "/science/LSAR/GSLC/metadata/sourceData/swaths/frequencyA/centerFrequency",
    "/science/LSAR/GSLC/metadata/sourceData/processingInformation/parameters/frequencyA/slantRange",
    "/science/LSAR/identification/productSpecificationVersion",
    "/science/LSAR/identification/productVersion",
    "/science/LSAR/identification/zeroDopplerEndTime",
    "/science/LSAR/identification/zeroDopplerStartTime",
    "/science/LSAR/identification/boundingPolygon",
    "/science/LSAR/identification/missionId",
    "/science/LSAR/identification/lookDirection",
    "/science/LSAR/identification/trackNumber",
    "/science/LSAR/identification/orbitPassDirection",
    "/science/LSAR/identification/absoluteOrbitNumber",
    "/science/LSAR/GSLC/metadata/orbit/orbitType",
]


def _convert_to_json_serializable(data):
    """Convert HDF5 data to JSON-serializable format.

    Parameters
    ----------
    data : any
        Data from HDF5 dataset (numpy arrays, scalars, strings, etc.)

    Returns
    -------
    any
        JSON-serializable version of the input data

    """
    if data is None:
        return None

    # Handle bytes
    if isinstance(data, bytes):
        return data.decode("utf-8")

    # Handle numpy scalar types (uint32, int64, float64, etc.)
    if isinstance(data, (np.integer, np.floating)):
        return data.item()

    # Handle numpy arrays
    if isinstance(data, np.ndarray):
        if data.size == 0:
            return []

        # String/bytes arrays
        if data.dtype.kind in ("U", "S", "O"):
            flat_list = []
            for item in data.flat:
                if isinstance(item, bytes):
                    flat_list.append(item.decode("utf-8"))
                elif isinstance(item, str):
                    flat_list.append(item)
                else:
                    flat_list.append(str(item))

            if data.size == 1:
                return flat_list[0]
            else:
                return np.array(flat_list).reshape(data.shape).tolist()

        # Numeric arrays - convert to list
        return data.tolist()

    # Handle lists/tuples recursively
    if isinstance(data, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in data]

    # Handle dictionaries recursively
    if isinstance(data, dict):
        return {
            key: _convert_to_json_serializable(value) for key, value in data.items()
        }

    # For Python native types (int, float, str, bool), return as-is
    return data


def _get_look_side_from_file(h5file: Filename) -> str:
    """Get the look side ("Left" / "Right") from a NISAR GSLC HDF5 file.

    Uses GDAL's multidim API so it works for ``/vsis3/...`` URLs as well as
    local paths.
    """
    for path in (
        "/science/LSAR/identification/lookDirection",
        "/identification/lookDirection",
    ):
        val = _read_string_mdarray(h5file, path)
        if val:
            return val.strip().rstrip("\x00") or "Right"
    return "Right"


def _extract_hdf5_metadata(h5file: Filename) -> dict:
    """Extract metadata datasets from a NISAR GSLC via GDAL's multidim API.

    Works for ``/vsis3/...`` URLs and local paths. Opens the file once and
    walks every path in ``NISAR_METADATA_PATHS`` against the same root
    group, so per-path cost is just the range reads, not a full reopen.
    """
    metadata: dict[str, Any] = {}
    if not HAS_GDAL:
        msg = "osgeo (GDAL) must be installed to use this function"
        raise ImportError(msg)

    ds = root = None
    try:
        ds = gdal.OpenEx(fspath(h5file), gdal.OF_MULTIDIM_RASTER)
        if ds is None:
            logger.debug(f"Could not open {h5file} with GDAL multidim API")
            return metadata
        root = ds.GetRootGroup()

        for dset_path in NISAR_METADATA_PATHS:
            try:
                value = _read_path_under_root(root, dset_path)
            except Exception as e:
                logger.debug(f"Could not extract {dset_path}: {e}")
                continue
            if value is None:
                continue
            try:
                metadata[dset_path] = _convert_to_json_serializable(value)
            except Exception as e:
                logger.debug(f"Could not serialize {dset_path}: {e}")
    finally:
        root = ds = None

    return metadata


def _read_path_under_root(root_group, dset_path: str):
    """Walk ``dset_path`` from ``root_group`` and return the MDArray value.

    Returns ``None`` if any intermediate group or the leaf MDArray is
    missing. Companion to :func:`_read_scalar_mdarray` for the case where
    you've already opened the file and want to amortize cost across many
    reads.
    """
    parts = [p for p in dset_path.split("/") if p]
    if not parts:
        return None
    grp = root_group
    ar = None
    try:
        for name in parts[:-1]:
            grp = grp.OpenGroup(name)
            if grp is None:
                return None
        ar = grp.OpenMDArray(parts[-1])
        if ar is None:
            return None
        return _read_mdarray_value(ar)
    finally:
        ar = None


def save_orbit_metadata_for_cslcs(
    cslc_files: list[Filename],
    subdataset: str | None,  # noqa: ARG001
    output_dir: Path,
) -> None:
    """Save orbit and metadata for all CSLC files for product creation.

    Saves:
    - Orbit data (times, positions, velocities)
    - Look direction
    - Zero doppler times
    - Orbit type and direction
    - All HDF5 metadata datasets needed for products

    Parameters
    ----------
    cslc_files : list[Filename]
        List of paths to CSLC/GSLC HDF5 files
    subdataset : str | None
        HDF5 subdataset path (if applicable)
    output_dir : Path
        Directory to save metadata cache files

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for cslc_file in cslc_files:
        # Skip compressed SLCs - they don't have orbit/metadata
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

            # Save orbit data to .npz (binary, fast)
            orbit_file = output_dir / f"orbit_{date_str}.npz"
            # Save metadata to .json (text, readable)
            metadata_file = output_dir / f"metadata_{date_str}.json"

            if orbit_file.exists() and metadata_file.exists():
                logger.debug(f"Metadata already cached for {date_str}")
                continue

            # Get orbit arrays from the file
            times, positions, velocities, reference_epoch = get_orbit_arrays(cslc_file)

            # Get look side
            look_side = _get_look_side_from_file(cslc_file)

            # Get zero doppler times
            zero_doppler_start = get_zero_doppler_time(
                cslc_file,
                dataset="/science/LSAR/identification/zeroDopplerStartTime",
                datetime_format="%Y-%m-%dT%H:%M:%S.%f",
            )
            zero_doppler_end = get_zero_doppler_time(
                cslc_file,
                dataset="/science/LSAR/identification/zeroDopplerEndTime",
                datetime_format="%Y-%m-%dT%H:%M:%S.%f",
            )

            # Extract HDF5 metadata datasets
            hdf5_metadata = _extract_hdf5_metadata(cslc_file)

            # Save orbit data (binary, fast access)
            np.savez_compressed(
                orbit_file,
                times=times,
                positions=positions,
                velocities=velocities,
                reference_epoch_iso=reference_epoch.isoformat(),
                look_side=look_side,
                source_file=str(cslc_file),
            )

            # Save metadata (text, readable)
            with open(metadata_file, "w") as f:
                json.dump(
                    {
                        "source_file": str(cslc_file),
                        "look_side": look_side,
                        "zero_doppler_start_time": zero_doppler_start.isoformat(),
                        "zero_doppler_end_time": zero_doppler_end.isoformat(),
                        "hdf5_datasets": hdf5_metadata,
                    },
                    f,
                    indent=2,
                )

            logger.debug(f"Saved metadata cache for {date_str}")

        except Exception as e:
            logger.warning(
                f"Failed to cache metadata for {cslc_file}: {e}. "
                "Product creation may fail if this file is needed."
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


def load_metadata(cache_dir: Path, cslc_filename: Filename) -> dict | None:
    """Load cached metadata for a CSLC file.

    Parameters
    ----------
    cache_dir : Path
        Directory containing cached metadata files
    cslc_filename : Filename
        Original CSLC filename (used to find the matching cache file)

    Returns
    -------
    dict | None
        Dictionary containing:
        - source_file: str
        - look_side: str
        - zero_doppler_start_time: str (ISO format)
        - zero_doppler_end_time: str (ISO format)
        - hdf5_datasets: dict mapping dataset paths to values
        Or None if not found

    """
    try:
        # Parse filename to find the matching metadata cache file
        parsed = parse_filename(cslc_filename)
        start_dt = parsed.get("start_datetime")
        if isinstance(start_dt, datetime):
            date_str = start_dt.strftime("%Y%m%dT%H%M%S")
        else:
            date_str = Path(cslc_filename).stem[:20]

        metadata_file = cache_dir / f"metadata_{date_str}.json"

        if not metadata_file.exists():
            # Try finding by glob pattern if exact match fails
            possible_files = list(cache_dir.glob(f"metadata_{date_str[:8]}*.json"))
            if possible_files:
                metadata_file = possible_files[0]
            else:
                logger.debug(f"Metadata cache file not found: {metadata_file}")
                return None

        # Load the metadata
        with open(metadata_file) as f:
            metadata = json.load(f)

        return metadata

    except Exception as e:
        logger.error(f"Failed to load metadata for {cslc_filename}: {e}")
        return None


def get_orbit_direction_from_cache(
    cache_dir: Path, cslc_filename: Filename
) -> str | None:
    """Get orbit direction from cache.

    Parameters
    ----------
    cache_dir : Path
        Directory containing cached metadata files
    cslc_filename : Filename
        Original CSLC filename

    Returns
    -------
    str | None
        "ascending" or "descending", or None if not found in cache

    """
    metadata = load_metadata(cache_dir, cslc_filename)
    if metadata is None:
        return None

    try:
        orbit_dir = metadata["hdf5_datasets"].get(
            "/science/LSAR/identification/orbitPassDirection"
        )
        return orbit_dir if orbit_dir else None
    except (KeyError, TypeError):
        return None


def get_orbit_type_from_cache(cache_dir: Path, cslc_filename: Filename) -> str | None:
    """Get orbit type from cache.

    Parameters
    ----------
    cache_dir : Path
        Directory containing cached metadata files
    cslc_filename : Filename
        Original CSLC filename

    Returns
    -------
    str | None
        Full orbit type name (e.g., "precise orbit Ephemeris"), or None if not found

    """
    orbit_types = {
        "POE": "precise orbit Ephemeris",
        "FOE": "Forecast Orbit Ephemeris",
        "NOE": "Near real-time Orbit Ephemeris",
        "MOE": "Medium precision Orbit Ephemeris",
        "DOE": "custom",
    }

    metadata = load_metadata(cache_dir, cslc_filename)
    if metadata is None:
        return None

    try:
        orbit_type_code = metadata["hdf5_datasets"].get(
            "/science/LSAR/GSLC/metadata/orbit/orbitType"
        )
        if orbit_type_code:
            return orbit_types.get(orbit_type_code)
        return None
    except (KeyError, TypeError):
        return None


def get_zero_doppler_time_from_cache(
    cache_dir: Path, cslc_filename: Filename, start_or_end: str = "start"
) -> datetime | None:
    """Get zero doppler time from cache.

    Parameters
    ----------
    cache_dir : Path
        Directory containing cached metadata files
    cslc_filename : Filename
        Original CSLC filename
    start_or_end : str
        Either "start" or "end"

    Returns
    -------
    datetime | None
        Zero doppler time, or None if not found

    """
    metadata = load_metadata(cache_dir, cslc_filename)
    if metadata is None:
        return None

    try:
        key = f"zero_doppler_{start_or_end}_time"
        time_str = metadata.get(key)
        if time_str:
            return datetime.fromisoformat(time_str)
        return None
    except (KeyError, TypeError, ValueError):
        return None


def copy_cached_metadata_to_file(
    cache_dir: Path,
    cslc_filename: Filename,
    output_file: Filename,
    dsets_to_copy: list[str],
    prepend_str: str = "",
) -> None:
    """Copy cached metadata to output HDF5 file.

    Parameters
    ----------
    cache_dir : Path
        Directory containing cached metadata files
    cslc_filename : Filename
        Original CSLC filename (used to find matching cache)
    output_file : Filename
        Path to output HDF5 file
    dsets_to_copy : list[str]
        List of HDF5 dataset paths to copy from cache
    prepend_str : str
        String to prepend to dataset names when copying (e.g., "reference_")

    """
    import h5py

    metadata = load_metadata(cache_dir, cslc_filename)
    if metadata is None:
        logger.warning(f"No cached metadata found for {cslc_filename}")
        return

    hdf5_datasets = metadata.get("hdf5_datasets", {})
    copied_count = 0

    with h5py.File(output_file, "a") as dst:
        for dset_path in dsets_to_copy:
            # Skip orbit groups - they have complex structure not cached
            if "orbit" in dset_path.lower() and dset_path.endswith("orbit"):
                logger.debug(
                    f"Skipping orbit group {dset_path} (complex structure, "
                    "orbit data available in .npz files)"
                )
                continue

            if dset_path not in hdf5_datasets:
                logger.debug(f"Dataset {dset_path} not in cache for {cslc_filename}")
                continue

            # Get the cached data
            data = hdf5_datasets[dset_path]

            # Create parent group if it doesn't exist
            out_group = str(Path(dset_path).parent)
            dst.require_group(out_group)

            # Create the dataset with prepended name if specified
            dset_name = Path(dset_path).name
            if prepend_str:
                dset_name = f"{prepend_str}{dset_name}"

            full_path = f"{out_group}/{dset_name}"

            # Delete if exists
            if full_path in dst:
                del dst[full_path]

            # Recreate the dataset
            try:
                if isinstance(data, str):
                    dst.create_dataset(full_path, data=np.bytes_(data))
                elif isinstance(data, list):
                    # Array data - reconstruct as numpy array
                    arr = np.array(data)
                    dst.create_dataset(full_path, data=arr)
                else:
                    # Scalar value
                    dst.create_dataset(full_path, data=data)
                copied_count += 1
            except Exception as e:
                logger.warning(f"Failed to copy {dset_path} to {full_path}: {e}")

    logger.debug(
        f"Copied {copied_count}/{len(dsets_to_copy)} datasets from cache to"
        f" {output_file}"
    )
