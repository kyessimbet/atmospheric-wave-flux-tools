

import xarray as xr
import netCDF4 as nc
import numpy as np
from epflux_module import compute_ep_flux

# Load dataset (replace with your own path)
dataset = xr.open_dataset('./data/winds_full_years/2007')

# variables
t = dataset.temperature                     # t(time,lat,lon,lev)
u = dataset.geopotential_height_geo_u_wind  # u wind
v = dataset.geopotential_height_geo_v_wind  # v wind
lat = np.array(dataset.latitude_bins)
lev = np.array(dataset.pressure / 100.)  # in hPa
time_dst = dataset.time

print(np.shape(t))

# Compute EP flux
F_merid, F_vert, F_div = compute_ep_flux(t, u, v, lat, lev)

# Save output to NetCDF
output = nc.Dataset('./2007_epflux.nc', 'w', format='NETCDF4')

output.createDimension("time", None)
output.createDimension("lat", None)
output.createDimension("lev", None)

t_var = output.createVariable('time', np.float32, ('time',))
lats = output.createVariable('lat', np.float32, ('lat',))
lev_var = output.createVariable('lev', np.float32, ('lev',))

Fphi_var = output.createVariable('Fphi', np.float64, ('time','lat','lev'))
Fp_var = output.createVariable('Fp', np.float64, ('time','lat','lev'))
Fdiv_var = output.createVariable('Fdiv', np.float64, ('time','lat','lev'))

t_var[:] = time_dst
lats[:] = lat
lev_var[:] = lev
Fphi_var[:] = F_merid
Fp_var[:] = F_vert
Fdiv_var[:] = F_div

# Units and metadata
Fphi_var.units = 'm^2/s^2'
Fp_var.units = 'm^2/s^2'
Fdiv_var.units = 'm/s^2'
lats.units = 'degrees north'
lev_var.units = 'hPa'

output.description = 'EP flux computed from winds and temperature'
output.close()

print('EP flux saved to 2007_epflux.nc')
