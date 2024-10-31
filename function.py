import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf  
import scipy
import matplotlib
from scipy import signal
import xarray as xr
import pandas as pd
import metpy.calc as mpcalc
from scipy import interpolate
from scipy.interpolate import interpn
import os
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from scipy import integrate
'''mask'''
def mask3D(da,mask_path,var_name):
    mask_daily=np.load(mask_path)
    #print(mask_daily.shape)
    mask_daily=np.expand_dims(mask_daily,0).repeat(len(da.time),axis=0)
    da_mask=da+mask_daily
    #plt.contourf(da_mask.data[1,:,:])
    #plt.title(var_name+' mask fig')
    return da_mask
def mask4D(da,mask_path,var_name):
    mask_daily=np.load(mask_path)
    #print(mask_daily.shape)
    mask_daily=np.expand_dims(mask_daily,0).repeat(len(da.time),axis=0)
    mask_daily_1ev=np.expand_dims(mask_daily,3).repeat(len(da.LEV),axis=3)
    da_mask=da+mask_daily_1ev
    return da_mask

'''对mask经向加权'''
def weight_lon(da,mask_path,var_name,color):
    if isinstance(mask_path, str):#mask_path的地方输入文件地址或者data array都可以
        u1=mask3D(da,mask_path,var_name)
    elif isinstance(mask_path, xr.DataArray):  # xarray 是一种用于处理 N 维标记数组的库
        u1=da
    weights = np.cos(np.deg2rad(u1.longitude))
    weights.name = "weights"
    u1_weighted = u1.weighted(weights)
    u1_weighted_lon_mean = u1_weighted.mean(("longitude"))
    u1_weighted_mean = u1_weighted.mean(("longitude","latitude"))
    #norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0,vmax=vmax)
    plt.contourf(u1_weighted_lon_mean.time.values,u1_weighted_lon_mean.latitude,u1_weighted_lon_mean.values.T,cmap=color)#,norm=norm)
    if len(u1_weighted_mean.time)>12:
        date = pd.date_range(u1_weighted_mean.time.data[0], periods=12,freq='30D')
        plt.xticks(date, ('1','2','3','4','5','6','7','8','9','10','11','12'))
    plt.colorbar()
    plt.ylim([0.5,19.5])
    plt.title(var_name+' time-lat fig')
    return u1_weighted_lon_mean
'''对mask纬向加权'''
def weight_lat(da,mask_path):
    if isinstance(mask_path, str):#mask_path的地方输入文件地址或者data array都可以
        u1=mask3D(da,mask_path,var_name)
    elif isinstance(mask_path, xr.DataArray):  # xarray 是一种用于处理 N 维标记数组的库
        u1=da
    weights = np.cos(np.deg2rad(u1.latitude))
    weights.name = "weights"
    u1_weighted = u1.weighted(weights)
    u1_weighted_lat_mean = u1_weighted.mean(("latitude"))
    plt.contourf(u1_weighted_lat_mean.time.values,u1_weighted_lat_mean.longitude,u1_weighted_lat_mean.values.T,cmap='jet')#,norm=norm)
    return u1_weighted_lat_mean

def weight_lon_and_lat4D(da,mask_path,var_name,color,lev_end,lat_min,lat_max,var_min, varmax, color_freq):
    if isinstance(mask_path, str):#mask_path的地方输入文件地址或者data array都可以
        u1=mask4D(da,mask_path,var_name)
    elif isinstance(mask_path, xr.DataArray):  # xarray 是一种用于处理 N 维标记数组的库
        u1=da#表示之前的数据已经进行过mask的操作，所以不需要mask处理了
    weights = np.cos(np.deg2rad(u1.longitude))
    weights.name = "weights"
    u1_weighted = u1.weighted(weights)
    u1_weighted_lon_mean = u1_weighted.mean(("longitude"))
    u1_weighted_mean=u1_weighted_lon_mean.sel(latitude=slice(lat_min,lat_max),LEV=slice(0,lev_end)).mean('latitude')
    user_response = input("是否画图? (请输入 yes 或 no): ")# Ask the user whether they want to draw the graph
    user_response = user_response.lower()# Normalize the user's response to lowercase for easier comparison
    if user_response == 'yes':# Check the user's response and act accordingly
        # Draw the graph using matplotlib  
        plt.contourf(u1_weighted_mean.time.values,u1_weighted_mean.LEV.values,u1_weighted_mean.data.T,cmap=color,levels=18)
        plt.gca().invert_yaxis()
        if len(u1_weighted_mean.time)>12:
            date = pd.date_range(u1_weighted_mean.time.data[0], periods=12,freq='30D')
            plt.xticks(date, ('1','2','3','4','5','6','7','8','9','10','11','12'))
        plt.colorbar(ticks=np.linspace(var_min, varmax, color_freq),label=var_name)
    elif user_response == 'no':
        print("已取消绘图操作.")
    else:
        print("无效输入. 请回答 'yes' 或 'no'.")
    return u1_weighted_mean

'''对mask经向和纬向加权'''
def weight_lon_and_lat(da,mask_path,var_name,latmin, latmax):
    if isinstance(mask_path, str):#mask_path的地方输入文件地址或者data array都可以
        u1=mask4D(da,mask_path,var_name)
    elif isinstance(mask_path, xr.DataArray):  # xarray 是一种用于处理 N 维标记数组的库
        u1=da#表示之前的数据已经进行过mask的操作，所以不需要mask处理了
    u1=u1.sel(latitude=slice(latmin, latmax))
    weights = np.cos(np.deg2rad(u1.longitude))
    weights.name = "weights"
    u1_weighted = u1.weighted(weights)
    u1_weighted_lon_mean = u1_weighted.mean(("longitude"))
    u1_weighted_mean = u1_weighted.mean(("longitude","latitude"))
    #加权与不加权平均的比较
    #plt.subplot(2,1,1)
    u1_weighted_mean.plot(label=var_name)
    #u1.mean(("longitude", "latitude")).plot(label=var_name+"_unweighted")
    plt.legend()
    #plt.show()
    return u1_weighted_mean

'''多年平均到一年'''
def years_to_year4D(da):
    group = da.groupby("time.year")##可以改成.month,还可以在后面加上.sum()or.mean()
    list(group)
    start_year=da.time.min().values.astype('datetime64[Y]').astype(int)+1970
    end_year=da.time.max().values.astype('datetime64[Y]').astype(int)+1970
    print(f"start year:{start_year},end year:{end_year}")
    #把不同年份相加
    sst_sum1=group[start_year].data*0
    a=0
    for i in range(start_year,end_year+1):
        sst_sum1=group[i].data+sst_sum1
        a=a+1
        #if i==2020:
            #print(i,sst_sum1.shape,a)
    #多年平均
    sst_avg=sst_sum1/a
    #创建dataset用来存储多年平均日data
    sst_avg = group[start_year]*0+sst_avg
    return sst_avg
def years_to_year3D(da,datetime):
    group = da.groupby("time.year")##可以改成.month,还可以在后面加上.sum()or.mean()
    list(group)
    start_year=da.time.min().values.astype('datetime64[Y]').astype(int)+1970
    end_year=da.time.max().values.astype('datetime64[Y]').astype(int)+1970
    print(f"start year:{start_year},end year:{end_year}")
    #把不同年份相加
    sst_sum1=np.zeros((len(datetime),len(da.latitude),len(da.longitude)))
    a=0;b=0
    for i in range(start_year,end_year+1):
        if yeardate(i)==365:
            sst_sum1=group[i].data+sst_sum1
            a=a+1
        else:
            '''关键'''
            sst_sum1=sst_sum1+np.delete(group[i], 59, 0).data
            a=a+1
            #if i==2020:
            #print(sst_sum1.shape,a)
    #多年平均
    sst_avg=sst_sum1/a+group[i]*0
    return sst_avg
#mld_avg=years_to_year3D(mld,1980,2021)
#mld_avg=xr.Dataset(data_vars={"mld": mld_avg })
#mld_avg=mld_avg.mld

'''去掉缺失值---可改进，改为补全平均值'''
def nan_to_0_4D(da):
    a00=da.sel(LEV=da.LEV[0]).data*0
    add_edge=np.expand_dims(a00,1).repeat(len(da.LEV),axis=1)
    da= da.fillna(0)+add_edge#把非边界区域的缺失值用0补全
    return da
def nan_to_0_3D(da):
    a00=da.sel(time=da.time[0])*0
    add_edge=np.expand_dims(a00,0).repeat(len(da.time),axis=0)
    da=da.fillna(0)+add_edge#把非边界区域的缺失值用0补全
    return da

'''加上表层数据并插值统一经纬度'''
def add_surface_samelatlon(param,path):
    ds0= xr.open_dataset(path)
    ds=ds0[param].interp(latitude=ds0.latitude.values, longitude=ds0.longitude.values)
    ds_add=ds.sel(LEV=ds.LEV[0])
    da = xr.concat([ds_add, ds], dim='LEV')
    LEV=np.append(0.0,ds0.LEV.data)#LEV纬度上增加0m
    #print(da.data.shape,LEV.shape)
    sst_avg = xr.DataArray(da, coords=[da.time,da.latitude,da.longitude,LEV], dims=['time', 'latitude','longitude','LEV'])
    return sst_avg

'''<-T_adv,-xadv,-yadv>'''
def cal_advection4D(t,u,v):
    '''ADV这里计算了负号'''
    ADV = np.zeros((len(t.time),len(t.LEV),len(t.latitude),len(t.longitude)))
    ADV_x= np.zeros((len(t.time),len(t.LEV),len(t.latitude),len(t.longitude)))
    ADV_y= np.zeros((len(t.time),len(t.LEV),len(t.latitude),len(t.longitude)))
    T_ADV= xr.DataArray(ADV, coords=[time,t.LEV,lat_loc,lon_loc], dims=['time','LEV', 'latitude','longitude'])
    ADV_x= xr.DataArray(ADV_x, coords=[time,t.LEV,lat_loc,lon_loc], dims=['time','LEV', 'latitude','longitude'])
    ADV_y= xr.DataArray(ADV_y, coords=[time,t.LEV,lat_loc,lon_loc], dims=['time','LEV', 'latitude','longitude'])
    for i in range(0,len(t.LEV)):
        T_ADV[:,i,:,:]=(-1)*mpcalc.advection(t.sel(LEV=t.LEV[i]).data, u=u.sel(LEV=t.LEV[i]), v=v.sel(LEV=t.LEV[i]))
        ADV_x[:,i,:,:]=(-1)*mpcalc.advection(t.sel(LEV=t.LEV[i]).data, u=u.sel(LEV=t.LEV[i]),v=0)#*second_of_day*(-1)* units('meter*K/sec')
        ADV_y[:,i,:,:]=(-1)*mpcalc.advection(t.sel(LEV=t.LEV[i]).data, u=0,v=v.sel(LEV=t.LEV[i]))
    ADV=xr.Dataset(data_vars={"T_ADV": T_ADV,'T_ADV_x':ADV_x,'T_ADV_y':ADV_y})
    plt.subplot(1,3,1)
    plt.contourf(t.longitude,t.latitude,ADV.T_ADV.data[1,1,:,:])
    plt.title('ADV')
    plt.subplot(1,3,2)
    plt.contourf(t.longitude,t.latitude,ADV.T_ADV_x.data[1,1,:,:])
    plt.title('x-ADV')
    plt.subplot(1,3,3)
    plt.contourf(t.longitude,t.latitude,ADV.T_ADV_y.data[1,1,:,:])
    plt.title('y-ADV')
    return ADV

'''画某几个月的图'''
def month_fig(da,startmonth,endmonth,plot_row,plot_colum,title,cmap_color):
    data=da
    import matplotlib.patches as patches
    fig,ax=plt.subplots(plot_row,plot_colum,dpi=100,figsize=(14, 10))
    ax = ax.flatten()
    '''https://blog.csdn.net/weixin_45727931/article/details/121961673'''
    for i in range(startmonth,endmonth+1):
        im=ax[i-startmonth].contourf(data.longitude,data.latitude ,data.data[i-1,:,:], levels=9,cmap=cmap_color)#,vmin = -140, vmax =160
        ax[i-startmonth].set_ylim([0,19.5])
        ax[i-startmonth].set_title(f'month-{i}'+title)
    cbar=fig.colorbar(im, ax=[ax[i] for i in range(endmonth-startmonth+1)], fraction=0.04, pad=0.05,orientation='horizontal')#fraction调大小
    #plt.savefig( 'E:\\data\\mask\\fig\\GODAS_12month_T-mean_C')
    plt.show()
    
'''插值和积分计算--mlw,mlv,mlu,mlt,[T],ADV=-[u*deltaT]注意带负号,0~mld!要把表层算进去！'''
##https://blog.csdn.net/weixin_47006934/article/details/107528265
def cal_dataset(d_dens,d_t,dens,t,u,v,w,adv,x_adv,y_adv):
    '''初始化'''
    lev=t.LEV
    print('t:',t.shape,'u:',u.shape,'v:',v.shape,'w:',w.shape,'dens:',dens.shape,'adv:',adv.shape)
    time=t.time
    lat_loc=t.latitude
    lon_loc=t.longitude
    mlu = np.zeros((len(time),len(lat_loc),len(lon_loc)))
    mlv = np.zeros((len(time),len(lat_loc),len(lon_loc)))
    mlw=np.zeros((len(time),len(lat_loc),len(lon_loc)))
    mlt = np.zeros((len(time),len(lat_loc),len(lon_loc)))
    T_mean = np.zeros((len(time),len(lat_loc),len(lon_loc)))
    ild = np.zeros((len(time),len(lat_loc),len(lon_loc)))
    mld= np.zeros((len(time),len(lat_loc),len(lon_loc)))
    mean_T_ADV=np.zeros((len(time),len(lat_loc),len(lon_loc)))
    x_mean_T_ADV=np.zeros((len(time),len(lat_loc),len(lon_loc)))
    y_mean_T_ADV=np.zeros((len(time),len(lat_loc),len(lon_loc)))
    '''设立维度--depth，lat，lon'''
    points=(w.latitude.values,w.longitude.values,w.LEV.values)
    for i in range(0,len(time)):
        #if i%10==0:
        print(i)
        for j in range(0,len(lat_loc)):
            for k in range(0,len(lon_loc)):#f是一个函数，用这个函数就可以找插值点的函数值了：
                #print(t.values[i,j,k,:].shape,lev.shape)
                f_ild=interpolate.interp1d(t.sel(time=time[i],latitude=lat_loc[j],longitude=lon_loc[k]),lev.values,kind='linear')
                f_mld=interpolate.interp1d(dens.sel(time=time[i],latitude=lat_loc[j],longitude=lon_loc[k]),lev.values,kind='linear')
                f_mlt=interpolate.interp1d(lev.values,t.sel(time=time[i],latitude=lat_loc[j],longitude=lon_loc[k]),kind='linear')
                f_u=interpolate.interp1d(lev.values,u.sel(time=time[i],latitude=lat_loc[j],longitude=lon_loc[k]),kind='linear')
                f_v=interpolate.interp1d(lev.values,v.sel(time=time[i],latitude=lat_loc[j],longitude=lon_loc[k]),kind='linear')
                f_adv=interpolate.interp1d(lev.values,adv.values[i,:,j,k],kind='linear')
                f_x_adv=interpolate.interp1d(lev.values,x_adv.values[i,:,j,k],kind='linear')
                f_y_adv=interpolate.interp1d(lev.values,y_adv.values[i,:,j,k],kind='linear')
                
                temp=t.sel(time=time[i],LEV=lev[0],latitude=lat_loc[j],longitude=lon_loc[k])
                ild[i,j,k]=f_ild(temp-d_t)#ild
                ssd=dens[i,j,k,0]
                mld[i,j,k]=f_mld(ssd+d_dens)
                depth=mld[i,j,k]#mld
                T_mean[i,j,k],err =integrate.quad(f_mlt,0,depth)/depth#<T>
                mlt[i,j,k]=f_mlt(depth)#mlt
                mlu[i,j,k]=f_u(depth)#mlu
                mlv[i,j,k]=f_v(depth)#mlv
                mean_T_ADV[i,j,k],err=integrate.quad(f_adv,0,depth) / (depth)#mean_T_ADV
                x_mean_T_ADV[i,j,k],err=integrate.quad(f_x_adv,0,depth)/(depth)#x_mean_T_ADV
                y_mean_T_ADV[i,j,k],err=integrate.quad(f_y_adv,0,depth)/(depth)#y_mean_T_ADV
                
                values=w.sel(time=w.time[i])#已知数据点的值，维度要和points一致
                xi=np.array([lat_loc[j],lon_loc[k],depth])#插值的坐标，维度要和points一致
                mlw[i,j,k]=interpn(points,values,xi, method='linear',fill_value=True, bounds_error=False)#mlw
                #外插时自动填充值，而不是nan # 允许外插
    '''dataset'''
    #print('ild:',ild.shape,'mld:',mld.shape,'mlt:',mlt.shape,'mlu:',mlu.shape,'mlv:',mlv.shape,'mlw:',mlw.shape)
    ild= xr.DataArray(ild, coords=[time,lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    mld= xr.DataArray(mld, coords=[time,lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    mlt= xr.DataArray(mlt, coords=[time,lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    mlu= xr.DataArray(mlu, coords=[time,lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    mlv= xr.DataArray(mlv, coords=[time,lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    mlw= xr.DataArray(mlw, coords=[time,lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    T_mean= xr.DataArray(T_mean, coords=[time,lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    mean_T_ADV= xr.DataArray(mean_T_ADV, coords=[time,lat_loc,lon_loc], dims=['time','latitude','longitude'])
    x_mean_T_ADV= xr.DataArray(x_mean_T_ADV, coords=[time,lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    y_mean_T_ADV= xr.DataArray(y_mean_T_ADV, coords=[time,lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    
    data_average=xr.Dataset(data_vars={'ild':ild,'mld':mld,'mlt': mlt,'mlu':mlu,'mlv':mlv,"mlw":mlw,'T_mean':T_mean,
                                       'T_ADV_mean':mean_T_ADV,'T_ADV_x_mean':x_mean_T_ADV,'T_ADV_y_mean':y_mean_T_ADV})
    return data_average
#https://blog.csdn.net/lanyuelvyun/article/details/120972936
def cal_mld_ild(d_dens,d_t,dens,t):
    '''初始化'''
    print('t:',t.shape,'dens:',dens.shape)
    lev=t.LEV
    time=t.time
    lat_loc=t.latitude
    lon_loc=t.longitude
    ild = np.zeros((len(time),len(lat_loc),len(lon_loc)))
    mld= np.zeros((len(time),len(lat_loc),len(lon_loc)))
    for i in range(0,len(time)):
        if i%10==0:
            print(i)
        for j in range(0,len(lat_loc)):
            for k in range(0,len(lon_loc)):#f是一个函数，用这个函数就可以找插值点的函数值了：
                f_ild=interpolate.interp1d(t.sel(time=time[i],latitude=lat_loc[j],longitude=lon_loc[k]),lev.values,
                                           kind='linear',fill_value="extrapolate")
                f_mld=interpolate.interp1d(dens.sel(time=time[i],latitude=lat_loc[j],longitude=lon_loc[k]),lev.values,
                                           kind='linear',fill_value="extrapolate")                
                temp0=t.sel(time=time[i],LEV=lev[0],latitude=lat_loc[j],longitude=lon_loc[k]);          #print(temp0.data)
                ild[i,j,k]=f_ild(temp0-d_t)#ild
                ssd=dens[i,j,k,0]
                mld[i,j,k]=f_mld(ssd+d_dens)#mld
    '''dataset'''
    ild= xr.DataArray(ild, coords=[time,lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    mld= xr.DataArray(mld, coords=[time,lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    data_average=xr.Dataset(data_vars={'ild':ild,'mld':mld})
    return data_average

'''tendency趋势'''
def tendency(da,da_name):
    '''tendency，需要在时间上进行一天的扩增，原因是月均数据在diff之后会只剩下11个，所以要扩增，方法就是把1月的数据增加一个到年1月'''
    append_da=da.sel(time=da.time[0])
    a00=np.expand_dims(append_da,0).repeat(1,axis=0)
    #第一个dim（0）表示在哪个轴上进行扩增，repeat（a,b）的a是扩增数量，axis的是沿哪个轴扩增
    '''创建dataset用来存储13个月的数据'''
    mld_add = xr.DataArray(a00, coords=[pd.date_range('2021-01-01', periods=1), lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    mld_add=xr.Dataset(data_vars={da_name: mld_add})
    mld_add=mld_add[da_name]
    merged_data = xr.concat([da, mld_add], dim='time')
    '''求偏导'''
    mld_tendency=np.diff(merged_data.data,axis=0)
    mld_tendency = xr.DataArray(mld_tendency, coords=[time, lat_loc,lon_loc], dims=['time', 'latitude','longitude'])
    return mld_tendency

'''沿时间合并文件'''
def conbinefile(filehead,startyear,endyear,filetail):
    ds0= xr.open_dataset(filehead + str(startyear)+ filetail)
    for year in range(startyear+1,endyear+1):
        da= xr.open_dataset(filehead + str(year)+ filetail)
        da=xr.concat([ds0, da], dim='time')
        ds0=da
    return da
'''判断闰年与否'''
def yeardate(year):
    if((year%4 == 0 and year%100 != 0)or (year % 400 == 0)):
        day=366
    else:
        day=365 
    return day
'''把每年每天数据加起来，去掉2-29'''
def daily_avg(da,startyear,endyear):
    group = da.groupby("time.year")
    data0 = group[startyear]*0
    a=0
    b=0
    for i in range(startyear,endyear+1):
        if yeardate(i)==365:
            data=group[i].data+data0
            data0=data
            a=a+1
        else:
            '''关键'''
            data = np.delete(group[i].data, 59, 0)#（数据，索引，纬度也即0或1或2）
            print(data.shape,i,group[i].shape)
            data=data+data0
            data0=data
            b=b+1
    sst_avg=data/(a+b)#多年平均
    sst_avg
    return sst_avg

'''制作mask'''
def matching_lat(lat_value,data): # 文件编码方式[lats,lons]，data:需要输入xr.open_dataset()
    '''经纬度与格点位置匹配,输入经纬度,nc数据,输出所在格点位置对应索引值 lon_idx：经度 lat_idx：纬度'''
    lats = data['latitude'].values# 获取纬度的坐标值
    lat_idx = np.abs(lats - lat_value).argmin()# 获取经纬度坐标的索引值
    return lat_idx
def matching_lon(lon_value,data): 
    lons = data['longitude'].values
    lon_idx = np.abs(lons - lon_value).argmin()
    return lon_idx
def make_mask4D(da,number_of_lon_to_save):
    data=da.sel(time=da.time[1],LEV=da.LEV[1])
    '''做一个框把海峡数据去掉'''
    box_lat_min=matching_lat(10,data)
    box_lat_max=matching_lat(data.latitude.data[-1],data)
    box_lon_min=matching_lon(data.longitude.data[0],data)
    box_lon_max=matching_lon(51,data)
    print(box_lat_min,box_lat_max,box_lon_min,box_lon_max)
    data.data[box_lat_min:box_lat_max,box_lon_min:box_lon_max]=np.nan
    plt.subplot(1,2,1)#类似matlab，1行二列的图，选择位置1存放，也可以用2行1列的subplot来存放
    plt.contourf(da.longitude,da.latitude,da.sel(time=da.time[1],LEV=da.LEV[1]),[250,300],levels=10,cmap='jet')
    plt.colorbar()
    '''去除离岸5°'''
    daily_mask=data*0#daily_mask代表只取离岸5°的数据
    for lat in range(0,len(data[:,0])):
        for lon in range(0,len(data[0,:])):
            if not np.isnan(daily_mask[lat,lon]):
                daily_mask[lat,lon+int(number_of_lon_to_save):len(data[0,:])]=np.nan
    plt.subplot(1,2,2)
    plt.contourf(da.longitude,da.latitude,da.sel(time=da.time[1],LEV=da.LEV[1])+daily_mask,[250,300],levels=10,cmap='jet')
    plt.colorbar()
    return daily_mask
def make_mask3D(da,number_of_lon_to_save,edge):
    data=edge.sel(time=edge.time[1])*0+da.sel(time=da.time[1])
    '''做一个框把海峡数据去掉'''
    box_lat_min=matching_lat(10,data)
    box_lat_max=matching_lat(data.latitude.data[-1],data)
    box_lon_min=matching_lon(data.longitude.data[0],data)
    box_lon_max=matching_lon(51,data)
    print(box_lat_min,box_lat_max,box_lon_min,box_lon_max)
    data.data[box_lat_min:box_lat_max,box_lon_min:box_lon_max]=np.nan
    plt.subplot(1,2,1)#类似matlab，1行二列的图，选择位置1存放，也可以用2行1列的subplot来存放
    plt.contourf(data.longitude,data.latitude,data,[250,300],levels=20,cmap='jet')
    plt.colorbar()
    '''去除离岸5°'''
    daily_mask=data.data+data*0#daily_mask代表只取离岸5°的数据
    #print(type(a))
    for lat in range(0,len(data.latitude)):
        for lon in range(0,len(data.longitude)):
            if not np.isnan(daily_mask[lat,lon]):
                daily_mask[lat,lon+int(number_of_lon_to_save):len(data[0,:])]=np.nan
    plt.subplot(1,2,2)
    plt.contourf(da.longitude,da.latitude,da.data[1,:,:]+daily_mask,[250,300],levels=20,cmap='jet')
    plt.colorbar()
    return daily_mask*0
