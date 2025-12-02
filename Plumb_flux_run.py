import xarray as xr
import numpy as np
import time
from netCDF4 import Dataset
from Plumb_flux import plumb_flux


in_file = "./data/Sept2019"
out_file = "./Plumb_flux/plumb_flux_output.nc"

ds = xr.open_dataset(in_file)

temp = ds["temperature"]
geop = ds["geopotential_height"]
lat = ds["latitude_bins"].values
lon = ds["longitude_bins"].values
lev = ds["pressure"].values
time_var = ds["time"].values

print("Temperature shape:", temp.shape)


Fx, Fy, Fz = plumb_flux(geop, lat, lon, lev, temp)


with Dataset(out_file, "w", format="NETCDF4") as f:
    f.createDimension("lat", len(lat))
    f.createDimension("lev", len(lev))
    f.createDimension("lon", len(lon))
    f.createDimension("time", None)

    f.createVariable("lat", "f4", ("lat",))[:] = lat
    f.createVariable("lon", "f4", ("lon",))[:] = lon
    f.createVariable("lev", "f4", ("lev",))[:] = lev / 100.0  # hPa
    f.createVariable("time", "f8", ("time",))[:] = time_var

    f.createVariable("Fx", "f4", ("time", "lev", "lat", "lon"))[:] = Fx
    f.createVariable("Fy", "f4", ("time", "lev", "lat", "lon"))[:] = Fy
    f.createVariable("Fz", "f4", ("time", "lev", "lat", "lon"))[:] = Fz

    f.description = "Plumb 3D flux (6-hourly)"
    f.history = "Created " + time.ctime()
    f.source = "xarray + netCDF4"

print(" Output written to", out_file)



