#!/usr/bin/env python
"""Convert disp-nisar displacement time series to velocity GeoTIFF.

This script reads OPERA DISP-NISAR displacement products (NetCDF format) and
computes a velocity map using linear regression across the time series. The
velocity is computed for each pixel and written to a GeoTIFF file.

Optionally converts line-of-sight (LOS) velocity to vertical velocity by
assuming all motion is vertical (zero horizontal velocity). This is a common
approximation in InSAR analysis.

Notes
-----
The script expects displacement filenames containing dates in YYYYMMDD format,
with the reference date and secondary date (e.g., disp_20200101_20200113.nc).
The reference date is automatically included as time zero with zero displacement.

Examples
--------
Compute LOS velocity from displacement time series:
    $ python displacement_to_velocity.py disp_*_*.nc -o velocity.tif

Compute vertical velocity assuming zero horizontal motion:
    $ python displacement_to_velocity.py disp_*_*.nc -o velocity.tif --vertical

Compute vertical velocity with custom incidence angle:
    $ python displacement_to_velocity.py disp_*_*.nc -o velocity.tif
             --vertical --incidence-angle 38.5

Save all outputs (velocity, R-squared, and intercept):
    $ python displacement_to_velocity.py disp_*_*.nc -o velocity.tif --save-intercept -v

"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.transform import Affine
from scipy import stats

logger = logging.getLogger(__name__)

DISPLACEMENT_DATASET = "/displacement"
TIME_DATASET = "/time"
SPATIAL_REF_DATASET = "/spatial_ref"
NEAR_INCIDENCE_ANGLE_DATASET = "/identification/near_range_incidence_angle"
FAR_INCIDENCE_ANGLE_DATASET = "/identification/far_range_incidence_angle"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.

    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "displacement_files",
        type=Path,
        nargs="+",
        help="Input displacement NetCDF files (OPERA DISP-NISAR products)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("velocity.tif"),
        help="Output velocity GeoTIFF file (default: velocity.tif)",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=np.nan,
        help="NoData value for output (default: NaN)",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=3,
        help="Minimum number of valid observations required per pixel (default: 3)",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help=(
            "Convert LOS velocity to vertical velocity assuming zero horizontal motion"
        ),
    )
    parser.add_argument(
        "--incidence-angle",
        type=float,
        default=None,
        help=(
            "Incidence angle in degrees for LOS to vertical conversion. If not"
            " provided, uses average of near/far range angles from the displacement"
            " file."
        ),
    )
    parser.add_argument(
        "--save-intercept",
        action="store_true",
        help="Save intercept from linear fit to separate GeoTIFF",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def parse_dates_from_filename(filename: Path) -> tuple[datetime, datetime]:
    """Parse reference and secondary dates from displacement filename.

    Expected format: *_refdate_secdate.nc or similar patterns with dates
    in YYYYMMDD format.

    Parameters
    ----------
    filename : Path
        Path to displacement file.

    Returns
    -------
    reference_date : datetime
        Reference date.
    secondary_date : datetime
        Secondary date.

    """
    stem = filename.stem
    # Try to find date patterns (YYYYMMDD format)
    import re

    date_pattern = r"(\d{8})"
    dates = re.findall(date_pattern, stem)

    if len(dates) < 2:
        raise ValueError(
            f"Could not parse reference and secondary dates from filename: {filename}. "
            "Expected format with two dates in YYYYMMDD format."
        )

    # Assume last two dates are reference and secondary
    ref_str, sec_str = dates[-2], dates[-1]
    reference_date = datetime.strptime(ref_str, "%Y%m%d")
    secondary_date = datetime.strptime(sec_str, "%Y%m%d")

    return reference_date, secondary_date


def read_displacement_stack(
    displacement_files: Sequence[Path],
) -> tuple[NDArray[np.float32], NDArray[np.float64], datetime]:
    """Read displacement data and times from multiple NetCDF files.

    Parses dates from filenames, sorts by secondary date, and includes
    reference date at time zero with zero displacement.

    Parameters
    ----------
    displacement_files : Sequence[Path]
        List of paths to displacement NetCDF files.

    Returns
    -------
    displacements : NDArray[np.float32]
        Displacement data with shape (n_dates, rows, cols).
        First layer is zeros (reference date).
    times : NDArray[np.float64]
        Time values in days since the reference date.
        First value is 0.0 (reference date).
    reference_date : datetime
        The reference date used for the time series.

    """
    logger.info(f"Reading {len(displacement_files)} displacement files")

    # Parse dates and sort by secondary date
    file_info = []
    for file_path in displacement_files:
        ref_date, sec_date = parse_dates_from_filename(file_path)
        file_info.append((file_path, ref_date, sec_date))

    # Sort by secondary date
    file_info.sort(key=lambda x: x[2])

    # Check that all files have the same reference date
    reference_dates = [info[1] for info in file_info]
    if len(set(reference_dates)) > 1:
        logger.warning(
            f"Multiple reference dates found: {set(reference_dates)}. "
            "Using the earliest as the reference."
        )
        reference_date = min(reference_dates)
    else:
        reference_date = reference_dates[0]

    logger.info(f"Reference date: {reference_date.strftime('%Y-%m-%d')}")

    displacements_list = []
    times_list = []

    # Read the first file to get the shape for the zero displacement at reference
    with h5py.File(file_info[0][0], "r") as f:
        shape = f[DISPLACEMENT_DATASET].shape

    # Add reference date with zero displacement
    displacements_list.append(np.zeros(shape, dtype=np.float32))
    times_list.append(0.0)

    for file_path, ref_date, sec_date in file_info:
        logger.debug(
            f"Reading {file_path.name}: {ref_date.strftime('%Y%m%d')} -> "
            f"{sec_date.strftime('%Y%m%d')}"
        )

        with h5py.File(file_path, "r") as f:
            # Read displacement data
            disp = f[DISPLACEMENT_DATASET][:]
            displacements_list.append(disp)

        # Compute time in days from reference date
        time_days = (sec_date - reference_date).total_seconds() / (24 * 3600)
        times_list.append(time_days)

    # Stack into arrays
    displacements = np.stack(displacements_list, axis=0)
    times_days = np.array(times_list)

    logger.info(f"Loaded displacement stack: {displacements.shape}")
    logger.info(f"Time range: {times_days.min():.1f} to {times_days.max():.1f} days")

    # Log the time series
    logger.info("Time series dates:")
    logger.info(f"  {reference_date.strftime('%Y-%m-%d')}: 0.0 days (reference)")
    for (file_path, _, sec_date), t in zip(file_info, times_days[1:]):
        logger.info(f"  {sec_date.strftime('%Y-%m-%d')}: {t:.1f} days")

    return displacements, times_days, reference_date


def read_geotransform(displacement_file: Path) -> tuple[Affine, CRS]:
    """Read the geotransform and CRS from a displacement file.

    Parameters
    ----------
    displacement_file : Path
        Path to a displacement NetCDF file.

    Returns
    -------
    transform : Affine
        Rasterio affine transform.
    crs : CRS
        Coordinate reference system.

    """
    with h5py.File(displacement_file, "r") as f:
        # Get GeoTransform string
        gt_string = f[SPATIAL_REF_DATASET].attrs["GeoTransform"]
        gt = [float(x) for x in gt_string.split()]

        # Convert to rasterio Affine transform
        transform = Affine.from_gdal(*gt)

        # Get CRS from spatial_ref attributes
        crs_wkt = f[SPATIAL_REF_DATASET].attrs["crs_wkt"]
        crs = CRS.from_wkt(crs_wkt)

    return transform, crs


def read_incidence_angle(displacement_file: Path) -> float:
    """Read the average incidence angle from a displacement file.

    Parameters
    ----------
    displacement_file : Path
        Path to a displacement NetCDF file.

    Returns
    -------
    float
        Average incidence angle in degrees.

    """
    with h5py.File(displacement_file, "r") as f:
        try:
            near_angle = f[NEAR_INCIDENCE_ANGLE_DATASET][()]
            far_angle = f[FAR_INCIDENCE_ANGLE_DATASET][()]
            avg_angle = (near_angle + far_angle) / 2.0
            logger.info(
                f"Read incidence angles: near={near_angle:.2f}°, far={far_angle:.2f}°, "
                f"average={avg_angle:.2f}°"
            )
            return float(avg_angle)
        except KeyError:
            logger.warning(
                "Could not find incidence angle datasets in file. "
                "Using default 40 degrees."
            )
            return 40.0


def convert_los_to_vertical(
    los_velocity: NDArray[np.float32],
    incidence_angle_deg: float,
) -> NDArray[np.float32]:
    """Convert LOS velocity to vertical velocity assuming zero horizontal motion.

    The conversion assumes that all motion is vertical and projects it
    onto the radar line of sight. The formula is:
        vertical_velocity = los_velocity / cos(incidence_angle)

    Parameters
    ----------
    los_velocity : NDArray[np.float32]
        Line-of-sight velocity in meters/year.
    incidence_angle_deg : float
        Incidence angle in degrees.

    Returns
    -------
    NDArray[np.float32]
        Vertical velocity in meters/year.

    References
    ----------
    This assumes the simplified geometry where horizontal velocity is zero:
        v_los = v_vertical * cos(incidence_angle)

    """
    incidence_angle_rad = np.deg2rad(incidence_angle_deg)
    vertical_velocity = los_velocity / np.cos(incidence_angle_rad)

    logger.info(
        f"Converted LOS to vertical using incidence angle {incidence_angle_deg:.2f}°"
    )
    logger.info(
        f"Vertical velocity range: {np.nanmin(vertical_velocity):.4f} to "
        f"{np.nanmax(vertical_velocity):.4f} m/yr"
    )

    return vertical_velocity


def compute_velocity(
    displacements: NDArray[np.float32],
    times: NDArray[np.float64],
    min_observations: int = 3,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Compute velocity from displacement time series using linear regression.

    Parameters
    ----------
    displacements : NDArray[np.float32]
        Displacement data with shape (n_dates, rows, cols).
    times : NDArray[np.float64]
        Time values in days since reference.
    min_observations : int
        Minimum number of valid observations required per pixel.

    Returns
    -------
    velocity : NDArray[np.float32]
        Velocity in meters/year with shape (rows, cols).
    intercept : NDArray[np.float32]
        Intercept of linear fit with shape (rows, cols).
    r_squared : NDArray[np.float32]
        R-squared value of linear fit with shape (rows, cols).

    """
    n_dates, rows, cols = displacements.shape
    logger.info(f"Computing velocity for {rows} x {cols} pixels")

    # Initialize output arrays
    velocity = np.full((rows, cols), np.nan, dtype=np.float32)
    intercept = np.full((rows, cols), np.nan, dtype=np.float32)
    r_squared = np.full((rows, cols), np.nan, dtype=np.float32)

    # Reshape for efficient computation
    disp_2d = displacements.reshape(n_dates, -1)  # (n_dates, n_pixels)

    # Find valid pixels (non-NaN)
    valid_mask = ~np.isnan(disp_2d)
    n_valid = valid_mask.sum(axis=0)  # Number of valid observations per pixel

    # Process only pixels with enough observations
    sufficient_data = n_valid >= min_observations
    n_pixels_processed = sufficient_data.sum()

    logger.info(
        f"Processing {n_pixels_processed:,} pixels with >= {min_observations}"
        " observations"
    )

    # Vectorized linear regression for all valid pixels
    for i in range(disp_2d.shape[1]):
        if not sufficient_data[i]:
            continue

        # Get valid data for this pixel
        mask = valid_mask[:, i]
        y = disp_2d[mask, i]
        x = times[mask]

        if len(x) < min_observations:
            continue

        # Linear regression: y = slope * x + intercept
        slope, intercept_val, r_value, _, _ = stats.linregress(x, y)

        # Convert slope from meters/day to meters/year
        velocity_val = slope * 365.25

        # Store results
        row_idx = i // cols
        col_idx = i % cols
        velocity[row_idx, col_idx] = velocity_val
        intercept[row_idx, col_idx] = intercept_val
        r_squared[row_idx, col_idx] = r_value**2

    logger.info(
        f"Velocity range: {np.nanmin(velocity):.4f} to {np.nanmax(velocity):.4f} m/yr"
    )

    return velocity, intercept, r_squared


def write_geotiff(
    output_path: Path,
    data: NDArray[np.float32],
    transform: Affine,
    crs: CRS,
    nodata: float = np.nan,
    description: str = "Velocity",
    units: str = "meters/year",
    reference_date: datetime | None = None,
) -> None:
    """Write data to a GeoTIFF file.

    Parameters
    ----------
    output_path : Path
        Path to output GeoTIFF file.
    data : NDArray[np.float32]
        Data array to write with shape (rows, cols).
    transform : Affine
        Rasterio affine transform.
    crs : CRS
        Coordinate reference system.
    nodata : float
        NoData value for output.
    description : str
        Description for the band.
    units : str
        Units for the data.
    reference_date : datetime, optional
        Reference date for the time series.

    """
    rows, cols = data.shape

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, description)
        tags = {"units": units}
        if reference_date is not None:
            tags["reference_date"] = reference_date.strftime("%Y-%m-%d")
        dst.update_tags(1, **tags)

    logger.info(f"Wrote {output_path}")


def main() -> None:
    """Run the displacement to velocity conversion."""
    args = parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate inputs
    if len(args.displacement_files) < 2:
        raise ValueError("At least 2 displacement files required to compute velocity")

    for f in args.displacement_files:
        if not f.exists():
            raise FileNotFoundError(f"File not found: {f}")

    # Read displacement stack
    displacements, times, reference_date = read_displacement_stack(args.displacement_files)

    # Compute velocity
    velocity, intercept, r_squared = compute_velocity(
        displacements,
        times,
        min_observations=args.min_observations,
    )

    # Read geospatial metadata from first file
    transform, crs = read_geotransform(args.displacement_files[0])

    # Convert to vertical velocity if requested
    if args.vertical:
        if args.incidence_angle is not None:
            incidence_angle = args.incidence_angle
            logger.info(f"Using user-specified incidence angle: {incidence_angle:.2f}°")
        else:
            incidence_angle = read_incidence_angle(args.displacement_files[0])

        # Save LOS velocity before conversion
        los_path = args.output.with_name(args.output.stem + "_los" + args.output.suffix)
        write_geotiff(
            los_path,
            velocity,
            transform,
            crs,
            nodata=args.nodata,
            description="Line-of-sight velocity",
            units="meters/year",
            reference_date=reference_date,
        )

        # Convert to vertical
        velocity = convert_los_to_vertical(velocity, incidence_angle)
        velocity_description = "Vertical velocity (assuming zero horizontal motion)"
    else:
        velocity_description = "Line-of-sight velocity"

    # Write outputs
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write velocity
    write_geotiff(
        args.output,
        velocity,
        transform,
        crs,
        nodata=args.nodata,
        description=velocity_description,
        units="meters/year",
        reference_date=reference_date,
    )

    # Write R-squared
    r2_path = args.output.with_suffix(".r2.tif")
    write_geotiff(
        r2_path,
        r_squared,
        transform,
        crs,
        nodata=args.nodata,
        description="R-squared of linear fit",
        units="unitless",
        reference_date=reference_date,
    )

    # Write intercept if requested
    if args.save_intercept:
        intercept_path = args.output.with_suffix(".intercept.tif")
        write_geotiff(
            intercept_path,
            intercept,
            transform,
            crs,
            nodata=args.nodata,
            description="Intercept of linear fit",
            units="meters",
            reference_date=reference_date,
        )

    logger.info("Velocity computation complete")


if __name__ == "__main__":
    main()
