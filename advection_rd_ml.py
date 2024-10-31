# ----------------------------------------------------------
# Calculate horizontal advection for mixed layer heat budget
# -----------------------------------------------------------
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import dask
import numpy.ma as ma
import datetime
import cftime

def advection_rd_ml(sal,lat,lon,time,Uh,Vh,st,end):
    # 1. Calculate horizontal tracer gradient
    RE=6378000
    latarr=lat*np.pi/180.;lonarr=lon*np.pi/180.;
    dSdx=sal.differentiate('lon',edge_order=1)*np.cos(lonarr)/lonarr.differentiate("lon",edge_order=1)
    dSdy=sal.differentiate("lat",edge_order=1)/latarr.differentiate("lat",edge_order=1)
    dSdx=dSdx/RE;dSdy=dSdy/RE
    
    # 2. Get climatological mean salinity and currents
    Sm=sal.groupby('time.month').mean()
    Um=Uh.groupby('time.month').mean()
    Vm=Vh.groupby('time.month').mean()
    # 3. Gradient of climatological mean salinity
    dS=Sm.differentiate('lon')
    dStmp=dS*np.cos(latarr.isel(lat=slice(0,-1)))/(lonarr.differentiate('lon'))
    dSmdx=0.5*(dStmp.isel(lon=slice(0,-1))+dStmp.isel(lon=slice(1,None)))/RE
    dStmp=(Sm.differentiate('lat'))/(latarr.differentiate('lat'))
    dSmdy=0.5*(dStmp.isel(lat=slice(0,-1))+dStmp.isel(lat=slice(1,None)))/RE

    arrU=Um*dSmdx; arrV=Vm*dSmdy
    mn_adv_clim_u=xr.DataArray(np.tile(arrU, (1, 1, 1)),coords=arrU.coords, dims=arrU.dims) # all nans yuck 
    mn_adv_clim_v=xr.DataArray(np.tile(arrV, (1, 1, 1)),coords=arrV.coords, dims=arrV.dims)
    # 4. Anomalous advection of climatological mean gradient
    anomadv_U=(Uh.groupby('time.month')-Um)*dSmdx#所有时间的距平
    anomadv_V=(Vh.groupby('time.month')-Vm)*dSmdy 
    ekman=anomadv_U + anomadv_V  # dims are fine
    # 5 Climatological mean advection of anomalous gradient
    climadv_U=Um*(dSdx.groupby('time.month') - dSmdx)
    climadv_V=Vm*(dSdy.groupby('time.month') - dSmdy)
    T1= climadv_U + climadv_V
    dSdx=dSdx.convert_calendar("proleptic_gregorian",use_cftime=False)
    dSdy=dSdy.convert_calendar("proleptic_gregorian",use_cftime=False)
    # 6 Anomalous advection of anomalous gradient
    anomadv_anomU=(Uh.groupby('time.month')-Um)*(dSdx.groupby('time.month') - dSmdx)
    anomadv_anomV=(Vh.groupby('time.month')-Vm)*(dSdy.groupby('time.month') - dSmdy)
    eddy1=anomadv_anomU + anomadv_anomV
    mneddy=eddy1.mean('time');h_eddy=eddy1+mneddy
    
    return mn_adv_clim_u, mn_adv_clim_v, ekman, T1, h_eddy#, dSpdt


