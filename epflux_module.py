
"""
Computes quasi-geostrophic Eliassen-Palm (EP) flux and its divergence
following Yessimbet et al. (2024).

This module provides functions to calculate the meridional and vertical
components of the EP flux and its divergence from temperature and wind
fields. .

Functions:
----------
- potential_temperature(T, lev): Compute potential temperature θ = T (p/p0)**κ
- zonal_mean_dev(var, dim='longitude_bins'): Compute deviation from zonal mean
- compute_ep_flux(t, u, v, lat, lev, scale=True): Compute EP flux and divergence
"""

import numpy as np
import xarray as xr
import netCDF4 as nc


def potential_temperature(T, lev):
    """Compute potential temperature θ = T (p/p0)**κ"""
    return T * (lev[np.newaxis, np.newaxis, np.newaxis, :] / 1000) ** (-0.286)

def zonal_mean_dev(var, dim='longitude_bins'):
    """Compute deviation from zonal mean along longitude"""
    return var - var.mean(dim=dim)

def compute_ep_flux(t, u, v, lat, lev, scale=True):
    """
    Quasi-geostrophic Eliassen-Palm (EP) flux and its divergence.

    Parameters
    ----------
    t : Temperature (time, lat, lon, lev)
    u : Zonal wind component
    v : Meridional wind component
    lat : np.ndarray
        Latitude array in degrees
    lev : np.ndarray
        Pressure levels in hPa
    scale : bool
        Whether to scale flux components

    Returns
    -------
    F_meridional, F_vertical, F_divergence
    """
    # constants
    a = 6.37122e06  # Earth's radius
    pilat = lat * np.pi / 180  # lats in radians
    coslat = np.cos(pilat)[np.newaxis, :, np.newaxis]  # geometric terms
    sinlat = np.sin(pilat)[np.newaxis, :, np.newaxis]
    omega = 7.2921e-5  # ω angular velocity of Earth's rotation rad/s
    f = 2 * omega * sinlat  # Coriolis parameter

    # Meridional component of EP flux, Fphi = − a cosφ ′u'′v'
    # First, compute ′u'′v'
    u_zonal_dev = zonal_mean_dev(u)
    v_zonal_dev = zonal_mean_dev(v)
    uv = u_zonal_dev * v_zonal_dev  # product of deviations
    uv_zonal_avg = uv.mean(['longitude_bins'])  # zonal average of the product
    F_meridional = -uv_zonal_avg * a * coslat * coslat  # meridional flux

    # Vertical component of EP flux, Fp = (f a cosφ ′v'′θ') / ′θp
    theta = potential_temperature(t, lev)  # potential temperature
    theta_zonal_avg = theta.mean(['longitude_bins'])
    loglev = np.log(lev)
    dp = np.gradient(loglev)[np.newaxis, np.newaxis, :]
    dthetadp = np.gradient(theta_zonal_avg, edge_order=2)[2] / dp
    dthetadp = dthetadp / (100. * lev[np.newaxis, np.newaxis, :])  # d′θ/dp
    theta_zonal_dev = theta - theta_zonal_avg
    vtheta = v_zonal_dev * theta_zonal_dev
    vtheta_zonal_avg = vtheta.mean(['longitude_bins'])
    F_vertical = f * a * coslat * vtheta_zonal_avg / dthetadp  # vertical flux

    # Divergence, Div = d(Fphi)/dphi + d(Fp)/dp
    dp = np.gradient(lev)[np.newaxis, np.newaxis, :]
    dphi = np.gradient(a * np.sin(pilat))[np.newaxis, :, np.newaxis]
    Div_Fphi = np.gradient(F_meridional, edge_order=2)[1] / dphi
    Div_Fp = np.gradient(F_vertical, edge_order=2)[2] / (dp * 100)
    F_divergence = Div_Fphi + Div_Fp  # total divergence
    Fdiv_zonal_a = F_divergence / (a * coslat)  # zonal acceleration

    # Scaling of vectors
    # first by coslat, Fphi has already additional coslat for displaying divergence
    # then both by sqrt(1000/p)
    # then Fphi/Pi and Fp/1.0e5
    if scale:
        F_meridional = F_meridional * (np.sqrt(1000. / lev)[np.newaxis, np.newaxis, :]) / (np.pi * a)
        F_vertical = F_vertical * coslat * (np.sqrt(1000. / lev)[np.newaxis, np.newaxis, :]) / 1.0e5

    return F_meridional, F_vertical, F_divergence
