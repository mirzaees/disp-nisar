import logging
from datetime import timedelta
from pathlib import Path

import h5py
import isce3
import numpy as np
from dolphin import baseline
from dolphin._types import Filename
from numpy.typing import ArrayLike
from pyproj import CRS, Transformer

logger = logging.getLogger(__name__)


def _get_look_side(h5file: Filename) -> isce3.core.LookSide:
    """Get the look side from a NISAR GSLC HDF5 file."""
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
                break
        else:
            # Default to right if not found
            return isce3.core.LookSide.Right

    if look_dir.lower() == "left":
        return isce3.core.LookSide.Left
    return isce3.core.LookSide.Right


def _get_grids(x: ArrayLike, y: ArrayLike, epsg: int) -> tuple:
    X, Y = np.meshgrid(x, y)
    xx = X.flatten()
    yy = Y.flatten()
    crs = CRS.from_epsg(epsg)
    utm_to_lonlat = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = utm_to_lonlat.transform(xx=xx, yy=yy, radians=False)
    lon = lon.reshape(X.shape)
    lat = lat.reshape(Y.shape)
    return lon, lat


def _load_orbit_from_cache(cache_dir: Path, cslc_filename: Filename) -> tuple:
    """Load orbit data from cache and reconstruct isce3.core.Orbit.

    Parameters
    ----------
    cache_dir : Path
        Directory containing cached orbit files
    cslc_filename : Filename
        Original CSLC filename (used to find matching cache)

    Returns
    -------
    tuple[isce3.core.Orbit, isce3.core.LookSide]
        Reconstructed orbit object and look side
    """
    from disp_nisar._orbit_cache import load_orbit_data

    orbit_data = load_orbit_data(cache_dir, cslc_filename)
    if orbit_data is None:
        raise FileNotFoundError(
            f"Could not load cached orbit data for {cslc_filename}. "
            f"Ensure orbit cache was generated at workflow start."
        )

    # Reconstruct isce3.core.Orbit from cached data
    times = orbit_data["times"]
    positions = orbit_data["positions"]
    velocities = orbit_data["velocities"]
    reference_epoch = orbit_data["reference_epoch"]

    orbit_svs = []
    for t, x, v in zip(times, positions, velocities):
        orbit_svs.append(
            isce3.core.StateVector(
                isce3.core.DateTime(reference_epoch + timedelta(seconds=float(t))),
                x,
                v,
            )
        )

    orbit = isce3.core.Orbit(orbit_svs)

    # Convert look side string to isce3.core.LookSide
    look_side_str = orbit_data["look_side"]
    if look_side_str.lower() == "left":
        side = isce3.core.LookSide.Left
    else:
        side = isce3.core.LookSide.Right

    return orbit, side


def compute_baselines(
    h5file_ref: Filename,
    h5file_sec: Filename,
    x: ArrayLike,
    y: ArrayLike,
    epsg: int,
    wavelength: float,
    height: float = 0.0,
    threshold: float = 1e-08,
    maxiter: int = 50,
    delta_range: float = 10.0,
    orbit_cache_dir: Path | None = None,
):
    """Compute the perpendicular baseline at a subsampled grid for two CSLCs.

    Parameters.
    ----------
    h5file_ref : Filename
        Path to reference OPERA NISAR CSLC HDF5 file.
    h5file_sec : Filename
        Path to secondary OPERA NISAR CSLC HDF5 file.
    height: float
        Target height to use for baseline computation.
        Default = 0.0
    latlon_subsample: int
        Factor by which to subsample the CSLC latitude/longitude grids.
        Default = 30
    threshold : float
        isce3 geo2rdr: azimuth time convergence threshold in meters
        Default = 1e-8
    maxiter : int
        isce3 geo2rdr: Maximum number of Newton-Raphson iterations
        Default = 50
    delta_range : float
        isce3 geo2rdr: Step size used for computing derivative of doppler
        Default = 10.0
    orbit_cache_dir : Path | None
        Directory containing cached orbit data. If provided, will load orbits
        from cache instead of accessing GSLC files.

    Returns
    -------
    baselines : np.ndarray
        2D array of perpendicular baselines

    """
    lon_grid, lat_grid = _get_grids(x=x, y=y, epsg=epsg)
    lon_arr = lon_grid.ravel()
    lat_arr = lat_grid.ravel()

    ellipsoid = isce3.core.Ellipsoid()
    zero_doppler = isce3.core.LUT2d()

    # Load orbit data from cache if available, otherwise from GSLC files
    if orbit_cache_dir is not None and orbit_cache_dir.exists():
        logger.info(f"Loading orbit data from cache: {orbit_cache_dir}")
        orbit_ref, side_ref = _load_orbit_from_cache(orbit_cache_dir, h5file_ref)
        orbit_sec, side_sec = _load_orbit_from_cache(orbit_cache_dir, h5file_sec)
        side = side_ref  # Use reference look side
    else:
        logger.info("Loading orbit data from GSLC files")
        from opera_utils import get_cslc_orbit
        side = _get_look_side(h5file_ref)
        orbit_ref = get_cslc_orbit(h5file_ref)
        orbit_sec = get_cslc_orbit(h5file_sec)

    baselines = []
    failed_count = 0

    for lon, lat in zip(lon_arr, lat_arr):
        llh_rad = np.deg2rad([lon, lat, height]).reshape((3, 1))

        try:
            # Try to convert geographic to radar coordinates
            az_time_ref, range_ref = isce3.geometry.geo2rdr(
                llh_rad,
                ellipsoid,
                orbit_ref,
                zero_doppler,
                wavelength,
                side,
                threshold=threshold,
                maxiter=maxiter,
                delta_range=delta_range,
            )
            az_time_sec, range_sec = isce3.geometry.geo2rdr(
                llh_rad,
                ellipsoid,
                orbit_sec,
                zero_doppler,
                wavelength,
                side,
                threshold=threshold,
                maxiter=maxiter,
                delta_range=delta_range,
            )

            pos_ref, velocity = orbit_ref.interpolate(az_time_ref)
            pos_sec, _ = orbit_sec.interpolate(az_time_sec)
            b = baseline.compute(
                llh_rad, pos_ref, pos_sec, range_ref, range_sec, velocity, ellipsoid
            )
            baselines.append(b)

        except RuntimeError as e:
            # geo2rdr failed to converge for this point (likely outside valid coverage)
            # Use NaN for this location and continue
            baselines.append(np.nan)
            failed_count += 1

    if failed_count > 0:
        logger.warning(
            f"Baseline computation: {failed_count}/{len(lon_arr)} points failed to "
            f"converge (likely outside valid swath coverage). Using NaN for these points."
        )

    return np.array(baselines).reshape(lon_grid.shape)


def _interpolate_data(
    data: np.ndarray, shape: tuple[int, int], method="linear"
) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    # Create coordinate arrays for the original data
    orig_coords = [np.linspace(0, 1, s) for s in data.shape]

    # Create coordinate arrays for the desired output shape
    new_coords = [np.linspace(0, 1, s) for s in shape]

    # Create the interpolator
    interp = RegularGridInterpolator(orig_coords, data, method=method)

    # Create a mesh grid for the new coordinates
    mesh = np.meshgrid(*new_coords, indexing="xy")

    # Perform the interpolation
    return interp(np.array(mesh).T.astype("float32"))
