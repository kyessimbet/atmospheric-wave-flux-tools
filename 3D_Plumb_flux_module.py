"""
Module for computing Plumb (1985) flux using xarray following Yessimbet et al. (2024).


This module calculates the Fx, Fy, and Fz components of the Plumb flux
from geopotential height and temperature fields. It includes utility functions
for computing potential temperature, Brunt–Väisälä frequency, Coriolis
parameter, and zonal mean removal.

Functions:
----------
- potential_temperature(temp, lev): Compute potential temperature θ = T (p/p0)**κ
- brunt_vaisala(theta, lev): Compute Brunt–Väisälä frequency squared N^2
- coriolis(lat): Compute Coriolis parameter f
- remove_zonal_mean(field, lon_dim='longitude_bins'): Subtract zonal mean
- plumb_flux(g, lat, lon, lev, temp): Compute Fx, Fy, Fz Plumb flux components 
"""

import numpy as np
import xarray as xr


def potential_temperature(temp, lev):
    """Compute potential temperature θ."""
    return temp * (1000.0 / lev) ** 0.286


def brunt_vaisala(theta, lev, sclhgt=8000.0, Rd=287.04):
    """Compute Brunt–Väisälä frequency squared (N^2)."""
    logp = np.log(lev / 1000.0)
    dthetadz = np.gradient(theta, -sclhgt * logp, axis=-1)
    return (Rd * (lev / 1000.0) ** 0.286) / sclhgt * dthetadz


def coriolis(lat, omega=7.2921e-5):
    """Coriolis parameter f."""
    return 2 * omega * np.sin(np.deg2rad(lat))


def remove_zonal_mean(field, lon_dim="longitude_bins"):
    """Remove zonal mean along longitude."""
    return field - field.mean(dim=lon_dim)


def plumb_flux(g, lat, lon, lev, temp):
    """
    Compute Plumb fluxes Fx, Fy, Fz.
    g: geopotential height
    lat, lon, lev: coordinates
    temp: temperature
    """
    a = 6.37122e6
    ga = 9.80665

    # reshape terms
    phi = np.deg2rad(lat)
    cos_phi = np.cos(phi)
    f = coriolis(lat)
    f = xr.DataArray(f, coords={"latitude_bins": lat}, dims=["latitude_bins"])

    lev_da = xr.DataArray(lev, dims=["pressure"])

    # potential temperature
    theta = potential_temperature(temp, lev_da)

    # Brunt–Väisälä
    N2 = brunt_vaisala(theta, lev)



    # derivatives
    dlon = np.gradient(lon)
    dlat = np.gradient(lat)
    dlev = np.gradient(lev)

    dpsi_lon = np.gradient(psi_dev, dlon, axis=psi_dev.dims.index("longitude_bins"))
    dpsi_lonlon = np.gradient(dpsi_lon, dlon, axis=psi_dev.dims.index("longitude_bins"))
    dpsi_lat = np.gradient(psi_dev, dlat, axis=psi_dev.dims.index("latitude_bins"))
    dpsi_lonlat = np.gradient(dpsi_lon, dlat, axis=psi_dev.dims.index("latitude_bins"))
    dpsi_z = np.gradient(psi_dev, dlev, axis=psi_dev.dims.index("pressure"))
    dpsi_lonz = np.gradient(dpsi_lon, dlev, axis=psi_dev.dims.index("pressure"))

    # fluxes
    Fx = (lev / 1000.0) / (2 * a**2 * cos_phi) * (dpsi_lon * dpsi_lon - psi_dev * dpsi_lonlon)
    Fy = (lev / 1000.0) / (2 * a**2) * (dpsi_lon * dpsi_lat - psi_dev * dpsi_lonlat)
    Fz = (f**2 * (lev / 1000.0)) / (2 * N2 * a) * (dpsi_lon * dpsi_z - psi_dev * dpsi_lonz)

    return Fx, Fy, Fz
