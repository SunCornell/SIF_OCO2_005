# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 22:02:46 2018

@author: shark
"""

import os
import gdal
import matplotlib.path as mplPath
import numpy as np
import copy
import netCDF4 as nc4
from netCDF4 import Dataset
import pyproj
import math
import pickle
import numpy.ma as ma
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPRegressor
import scipy
#please adjust the file systerm before running
#This script is for every 16 days prediciton
#here take the first 16 days of August,2015 as an example
################################################################################
#extracting training samples
names = locals()
yearid=2015
monthid=8
pn=1# 201508a: first 16 days of 201508
if monthid<10:
    monthstr="0"+str(monthid)
    month=str(yearid)+"/"+monthstr+"/"
else:
    monthstr=str(monthid)
    month=str(yearid)+"/"+monthstr+"/"
###########################
#columns: subtype1,subtype2,minlon,maxlon,minlat,maxlat
model_array=np.array([[1,3,-180,180,0,90],[1,3,-180,180,-90,0],\
                      [2,0,-180,-25,-90,90],[2,0,-25,60,-90,90],[2,0,60,180,-90,90],
                      [4,5,-180,180,0,90],[4,5,-180,180,-90,0],[6,7,-180,180,0,90],\
                      [6,7,-180,180,-90,0],[8,9,-180,180,0,90],[8,9,-180,180,-90,0],\
                      [10,0,-180,180,0,90],[10,0,-180,180,-90,0],[12,14,-180,-25,0,90],\
                      [12,14,-180,-25,-90,0],[12,14,-25,60,0,90],[12,14,-25,60,-90,0],\
                      [12,14,60,180,0,90],[12,14,60,180,-90,0]])
###########################
path='/local1/storage/data/OCO2/sif_lite_B8100/'+month
files= os.listdir(path)
datelst=np.arange(len(files))
datestrlst=[]
for i in np.arange(len(files)):
    filename=files[i]
    idnum=filename.split('_')
    keyname=idnum[2]
    datestr=keyname[4:]
    datestrlst.append(datestr)
    datename=int(keyname[4:])
    datelst[i]=datename

maxday=[31,28,31,30,31,30,31,31,30,31,30,31]

oco2lst=[]
if pn == 1:
    for i in np.arange(len(datelst)):
        if datelst[i]<=16:
            oco2lst.append(files[i])
if pn == 2:
    for i in np.arange(len(datelst)):
        if (datelst[i]<=maxday[monthid-1])&(datelst[i]>=(maxday[monthid-1]-15)):
            oco2lst.append(files[i])
# create CoordinateTransformation by pyproj
wcg84=pyproj.Proj("+init=EPSG:4326") 
sinu=pyproj.Proj("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
############################
for i in np.arange(len(datelst)):
    print datelst[i]
    if monthid<10:
        path101 = '/local1/storage/data/BRDF2015/e4ftl01.cr.usgs.gov/MOTA/MCD43A4.006/'+str(yearid)+".0"+str(monthid)+"."+datestrlst[i]
    else:
        path101 = '/local1/storage/data/BRDF2015/e4ftl01.cr.usgs.gov/MOTA/MCD43A4.006/'+str(yearid)+"."+str(monthid)+"."+datestrlst[i]
    files101= os.listdir(path101) #obtain all file names in folder 
    dic101 = {}  
    for file in files101: #Traversing folder
        if not os.path.isdir(file):
            idnum=file.split('.')
            keyname=idnum[2]
            modis = gdal.Open(path101+"/"+file)# hdf_ds should now be a GDAL dataset, but if the file isn't found
            subdatasets = modis.GetSubDatasets()
            sds = gdal.Open(modis.GetSubDatasets()[0][0])
            mb=sds.ReadAsArray()
            [m,n]=mb.shape
            ## get the projection
            proj1 = sds.GetProjection()
            geotransform = sds.GetGeoTransform()
    
            originX1 = geotransform[0] 
            originY1 = geotransform[3]
            pixelWidth1 = geotransform[1] 
            pixelHeight1 = geotransform[5] 
    
            lonleft=originX1
            lattop=originY1
            lonright=lonleft+m*pixelWidth1
            latbot=lattop+n*pixelHeight1
            
            [wlon0,wlat0]=pyproj.transform(sinu,wcg84,lonleft, latbot)
            [wlon1,wlat1]=pyproj.transform(sinu,wcg84,lonleft, lattop)
            [wlon2,wlat2]=pyproj.transform(sinu,wcg84,lonright, lattop)
            [wlon3,wlat3]=pyproj.transform(sinu,wcg84,lonright, latbot)          
            if (wlon0>wlon3)or(wlon0>wlon2)or(wlon1>wlon3)or(wlon1>wlon2):
                if wlon0>170:
                    wlon0=-360+wlon0
                if wlon1>170:
                    wlon1=-360+wlon1    
                if wlon2>170:
                    wlon2=-360+wlon2
                if wlon3>170:
                    wlon3=-360+wlon3
            dic101[file]=[lonleft,lonright,latbot,lattop,wlon0,wlat0,wlon1,wlat1,\
            wlon2,wlat2,wlon3,wlat3,pixelWidth1,pixelHeight1,m,n,keyname]

            names['dic1'+str(int(datelst[i]))]=dic101
            f = open('dic1'+str(int(datelst[i]))+'.txt','w')
            pickle.dump(dic101,f)
            f.close()
#################
for ik in np.arange(len(oco2lst)):
#for ik in [0,1]:
    oco2_d01=oco2lst[ik] 
    print oco2_d01
    print 'reading oco2_',ik
    oco = Dataset(path+oco2_d01)
    lons = oco.variables['longitude'][:]
    lats = oco.variables['latitude'][:]
    flons=oco.variables['footprint_vertex_longitude'][:]
    flats=oco.variables['footprint_vertex_latitude'][:]
    sif757=oco.variables['SIF_757nm'][:]
    sif771=oco.variables['SIF_771nm'][:]
    lcsif=oco.variables['IGBP_index'][:]
    sza=oco.variables['solar_zenith_angle'][:]
    dcf=oco.variables['daily_correction_factor'][:]
    mm=oco.variables['measurement_mode'][:]
    
    
    lons_d01=lons[mm==0]
    lats_d01=lats[mm==0]
    flons_d01=flons[mm==0]
    flats_d01=flats[mm==0]
    sif757_d01=sif757[mm==0]
    sif771_d01=sif771[mm==0]
    lcsif_d01=lcsif[mm==0]
    dcf_d01=dcf[mm==0]
    sza_d01=sza[mm==0]
    
    l_d01=len(lons_d01)
    dcsif757_d01=sif757_d01*dcf_d01
    dcsif771_d01=sif771_d01*dcf_d01
    
    COSsza_d01=np.zeros(l_d01)
    for i in np.arange(l_d01):
        COSsza_d01[i]=round(math.cos(math.radians(sza_d01[i])),6)
    
    COSsif757_d01=sif757_d01/COSsza_d01 
    COSsif771_d01=sif771_d01/COSsza_d01
      
    day_d01=np.ones((l_d01))*int(oco2_d01[15:17])
    
    date_d01=int('20'+oco2_d01[11:17])*np.ones((l_d01))
    
    Asif757_d01=copy.copy(sif757_d01)
    Asif771_d01=copy.copy(sif771_d01)
    lctype=np.arange(15)
    lc_type=lctype[1:]
    idx_all=np.arange(l_d01)    
    ##############################
    a=np.zeros((l_d01,5))
    nidx_d01=a.astype(int)   
    ##############################
    for lc in lc_type:
        idx_lc=idx_all[lcsif_d01==lc]
        lon_lc=lons_d01[idx_lc]
        lat_lc=lats_d01[idx_lc]
        num_data=len(idx_lc)
        if num_data>11: 
            idx2=np.arange(num_data)
            for i1 in idx2:
                lon=lon_lc[i1]
                lat=lat_lc[i1]
                diff_lons=lon_lc-lon*(np.ones(num_data))
                diff_lats=lat_lc-lat*(np.ones(num_data))
                dist_center=diff_lons**2+diff_lats**2
                sort_index = np.argsort(dist_center)        
                idx_sorted=idx_lc[sort_index]           
                Neighbors1=idx_sorted[:5]
                
                i_all=idx_lc[i1]        
                Asif757_d01[i_all]=np.mean(sif757_d01[Neighbors1])
                Asif771_d01[i_all]=np.mean(sif771_d01[Neighbors1])
                ####
                nidx_d01[i_all]=Neighbors1
                ####
    Adcsif757_d01=Asif757_d01*dcf_d01
    Adcsif771_d01=Asif771_d01*dcf_d01            
                
    ACOSsif757_d01=Asif757_d01/COSsza_d01 
    ACOSsif771_d01=Asif771_d01/COSsza_d01  
    if ik==0:
        lonsf=lons_d01
        latsf=lats_d01
        flonsf=flons_d01
        flatsf=flats_d01
        dayf=day_d01
        datef=date_d01
        lcsiff=lcsif_d01
        sif757f=sif757_d01
        sif771f=sif771_d01
        dcsif757f=dcsif757_d01
        dcsif771f=dcsif771_d01
        Asif771f=Asif771_d01
        Asif757f=Asif757_d01
        COSsif757f=COSsif757_d01
        COSsif771f=COSsif771_d01
        ACOSsif757f=ACOSsif757_d01
        ACOSsif771f=ACOSsif771_d01
        Adcsif757f=Adcsif757_d01
        Adcsif771f=Adcsif771_d01
        
        nidxf=nidx_d01
    else:
        lonsf=np.concatenate([lonsf, lons_d01])
        latsf=np.concatenate([latsf, lats_d01])
        flonsf=np.concatenate([flonsf, flons_d01])
        flatsf=np.concatenate([flatsf, flats_d01])
        dayf=np.concatenate([dayf, day_d01])
        datef=np.concatenate([datef, date_d01])
        lcsiff=np.concatenate([lcsiff, lcsif_d01])
        sif757f=np.concatenate([sif757f, sif757_d01])
        sif771f=np.concatenate([sif771f, sif771_d01])
        dcsif757f=np.concatenate([dcsif757f, dcsif757_d01])
        dcsif771f=np.concatenate([dcsif771f, dcsif771_d01])
        Asif771f=np.concatenate([Asif771f, Asif771_d01])
        Asif757f=np.concatenate([Asif757f, Asif757_d01])
        COSsif757f=np.concatenate([COSsif757f, COSsif757_d01])
        COSsif771f=np.concatenate([COSsif771f, COSsif771_d01])
        ACOSsif757f=np.concatenate([ACOSsif757f, ACOSsif757_d01])
        ACOSsif771f=np.concatenate([ACOSsif771f, ACOSsif771_d01])
        Adcsif757f=np.concatenate([Adcsif757f, Adcsif757_d01])
        Adcsif771f=np.concatenate([Adcsif771f, Adcsif771_d01])
    #############
        nidx_d012=nidx_d01+len(nidxf)
        print len(nidxf)
        np.save('nidx'+str(ik)+'.npy',nidx_d012)
        nidxf=np.concatenate([nidxf, nidx_d012])
    ##########################         
np.save('lons.npy',lonsf)
np.save('lats.npy',latsf)
np.save('flons.npy',flonsf)
np.save('flats.npy',flatsf)
np.save('lcsif.npy',lcsiff)
np.save('day.npy',dayf)
np.save('date.npy',datef)

np.save('sif757.npy',sif757f)
np.save('sif771.npy',sif771f)
np.save('dcsif757.npy',dcsif757f)
np.save('dcsif771.npy',dcsif771f)
np.save('COSsif757.npy',COSsif757f)
np.save('COSsif771.npy',COSsif771f)
np.save('Asif757.npy',Asif757f)
np.save('Asif771.npy',Asif771f)
np.save('ACOSsif757.npy',ACOSsif757f)
np.save('ACOSsif771.npy',ACOSsif771f)
np.save('Adcsif757.npy',Adcsif757f)
np.save('Adcsif771.npy',Adcsif771f)
#########
np.save('nidx.npy',nidxf)
##########
lons=np.load('/local/workdir/Yu/workdir1015/201508a/data/lons.npy')
lats=np.load('/local/workdir/Yu/workdir1015/201508a/data/lats.npy')
flons=np.load('/local/workdir/Yu/workdir1015/201508a/data/flons.npy')
flats=np.load('/local/workdir/Yu/workdir1015/201508a/data/flats.npy')
lcsif=np.load('/local/workdir/Yu/workdir1015/201508a/data/lcsif.npy')
day=np.load('/local/workdir/Yu/workdir1015/201508a/data/day.npy')
date=np.load('/local/workdir/Yu/workdir1015/201508a/data/date.npy')

sif757=np.load('/local/workdir/Yu/workdir1015/201508a/data/sif757.npy')
sif771=np.load('/local/workdir/Yu/workdir1015/201508a/data/sif771.npy')
dcsif757=np.load('/local/workdir/Yu/workdir1015/201508a/data/dcsif757.npy')
dcsif771=np.load('/local/workdir/Yu/workdir1015/201508a/data/dcsif771.npy')
COSsif757=np.load('/local/workdir/Yu/workdir1015/201508a/data/COSsif757.npy')
COSsif771=np.load('/local/workdir/Yu/workdir1015/201508a/data/COSsif771.npy')
Asif757=np.load('/local/workdir/Yu/workdir1015/201508a/data/Asif757.npy')
Asif771=np.load('/local/workdir/Yu/workdir1015/201508a/data/Asif771.npy')
Adcsif757=np.load('/local/workdir/Yu/workdir1015/201508a/data/Adcsif757.npy')
Adcsif771=np.load('/local/workdir/Yu/workdir1015/201508a/data/Adcsif771.npy')
ACOSsif757=np.load('/local/workdir/Yu/workdir1015/201508a/data/ACOSsif757.npy')
ACOSsif771=np.load('/local/workdir/Yu/workdir1015/201508a/data/ACOSsif771.npy')
######
nidx=np.load('/local/workdir/Yu/workdir1015/201508a/data/nidx.npy')

total_num=len(lons)
print 'totalnum:%d'% total_num
total_idx=np.arange(total_num)

num_model=19
for mn in np.arange(num_model):
    lc1=model_array[mn,0]
    lc2=model_array[mn,1]
    lon1=model_array[mn,2]
    lon2=model_array[mn,3]    
    lat1=model_array[mn,4]
    lat2=model_array[mn,5]    
     
    if lc2==0:
        idx0=np.arange(total_num)
        idx=idx0[(lcsif==lc1)&(lons>lon1)&(lons<lon2)&(lats>lat1)&(lats<lat2)]
        
    elif lc2>0:
        idx0=np.arange(total_num)
        idxa=idx0[(lcsif==lc1)&(lons>lon1)&(lons<lon2)&(lats>lat1)&(lats<lat2)]
        idxb=idx0[(lcsif==lc2)&(lons>lon1)&(lons<lon2)&(lats>lat1)&(lats<lat2)]
        idx=np.concatenate([idxa,idxb])
        
    lons_LC=lons[idx]
       
    LC_num=len(lons_LC)
    
    idx_random=np.random.permutation(len(idx))
    idx=idx[idx_random]
    
    train_num=2000
    if train_num>LC_num:
        train_num=LC_num
     
    dataset_all=np.zeros((train_num,52))
           
    i2=0
    b=np.zeros((2,5))
    sort_container=b.astype(int)   
    for ia in idx:    
        if i2 <train_num: 
            
            nidxi=nidx[ia]
            if (np.count_nonzero(nidxi)!=0):
                nidxi_sort=nidxi[np.argsort(nidxi)]
                nidxi_sort=nidxi_sort.reshape(1,5)
                if nidxi_sort in sort_container:
                    c=0
                else:
                    sort_container=np.concatenate([sort_container,nidxi_sort])
                        
                    vc=np.zeros((5,14))
                    nid=0
                    for i in nidxi:
                                 
                        floni=flons[i,:]
                        flati=flats[i,:]
                        
                        lon0=floni[0]
                        lon1=floni[1]
                        lon2=floni[2]
                        lon3=floni[3]
                        
                        lat0=flati[0]
                        lat1=flati[1]
                        lat2=flati[2]
                        lat3=flati[3]
                                            
                        loni=lons[i]
                        lati=lats[i]
                    
                        lc_oco=lcsif[i]
                        date_v=date[i]
                        day_v=day[i]
                        sif757_v=sif757[i]
                        dcsif757_v=dcsif757[i]
                        COSsif757_v=COSsif757[i]
                        Asif757_v=Asif757[i]
                        Adcsif757_v=Adcsif757[i]
                        ACOSsif757_v=ACOSsif757[i]    
                        sif771_v=sif771[i]
                        dcsif771_v=dcsif771[i]
                        COSsif771_v=COSsif771[i]
                        Asif771_v=Asif771[i]
                        Adcsif771_v=Adcsif771[i]
                        ACOSsif771_v=ACOSsif771[i]     
                  
                        [modis_x, modis_y] = pyproj.transform(wcg84,sinu,loni, lati)
                                        
                        dic1=names['dic1'+str(int(day_v))]

                        i_break=0
                        for key in dic1.keys():
                            geoinfo=dic1[key]
                            #[lonleft,lonright,latbot,lattop,4:wlon0,5:wlat0,wlon1,wlat1,\
                                    #wlon2,wlat2,10:wlon3,wlat3,pixelWidth1,pixelHeight1,m,n,keyname]
                            [wlon0,wlat0,wlon1,wlat1,wlon2,wlat2,wlon3,wlat3,m,n,keyname]=\
                            (geoinfo[4],geoinfo[5],geoinfo[6],geoinfo[7],geoinfo[8],geoinfo[9],\
                             geoinfo[10],geoinfo[11],geoinfo[14],geoinfo[15],geoinfo[16])
                            
                            m_vet = [[wlon0,wlat0], [wlon1,wlat1], [wlon2, wlat2], [wlon3, wlat3]]
                            mrange = mplPath.Path(np.array(m_vet))
                                      
                            if (mrange.contains_point((lon0,lat0)))&(mrange.contains_point((lon1,lat1)))&\
                            (mrange.contains_point((lon2,lat2)))&(mrange.contains_point((lon3,lat3))):
                            
                                originX=geoinfo[0]
                                originY=geoinfo[3]
                                pixelWidth=geoinfo[12]
                                pixelHeight=geoinfo[13]
                                
                                xOffset = int((modis_x - originX)/pixelWidth)
                                yOffset = int((modis_y - originY)/pixelHeight)
                               
                                minxoffset=xOffset-2
                                maxxoffset=xOffset+2
                                minyoffset=yOffset-3
                                maxyoffset=yOffset+3
                                     
                                if (minxoffset>0)&(maxxoffset<n)&(minyoffset>0)&(maxyoffset<m):
                                    
                                    poly_vet = [[lon0,lat0], [lon1,lat1], [lon2, lat2], [lon3, lat3]]
                                    footprint = mplPath.Path(np.array(poly_vet))
                                    
                                    if day_v<10:#/local1/storage/data/MODIS_BRDF/e4ftl01.cr.usgs.gov/MOTA/MCD43A4.006/
                                        path1='/local1/storage/data/BRDF2015/e4ftl01.cr.usgs.gov/MOTA/MCD43A4.006/'+str(yearid)+"."+monthstr+".0"+str(int(day_v))
                                    else:
                                        path1='/local1/storage/data/BRDF2015/e4ftl01.cr.usgs.gov/MOTA/MCD43A4.006/'+str(yearid)+"."+monthstr+"."+str(int(day_v))
                                    
                                    modis1 = gdal.Open(path1+"/"+key)
                                    subdatasets1 = modis1.GetSubDatasets()
                                    
                                    sds_m1b1 = gdal.Open(modis1.GetSubDatasets()[7][0])
                                    b1=sds_m1b1.ReadAsArray()
                                
                                    sds_m1b2 = gdal.Open(modis1.GetSubDatasets()[8][0])
                                    b2=sds_m1b2.ReadAsArray()
                                
                                    sds_m1b3 = gdal.Open(modis1.GetSubDatasets()[9][0])
                                    b3=sds_m1b3.ReadAsArray()
                                
                                    sds_m1b4 = gdal.Open(modis1.GetSubDatasets()[10][0])
                                    b4=sds_m1b4.ReadAsArray()
                                
                                    sds_m1b5 = gdal.Open(modis1.GetSubDatasets()[11][0])
                                    b5=sds_m1b5.ReadAsArray()
                                
                                    sds_m1b6 = gdal.Open(modis1.GetSubDatasets()[12][0])
                                    b6=sds_m1b6.ReadAsArray()
                                
                                    sds_m1b7 = gdal.Open(modis1.GetSubDatasets()[13][0])
                                    b7=sds_m1b7.ReadAsArray()
                                    
                                    sds_m1b1 = gdal.Open(modis1.GetSubDatasets()[0][0])
                                    quality=sds_m1b1.ReadAsArray()
                    
                                    sb1=0
                                    sb2=0
                                    sb3=0
                                    sb4=0
                                    sb5=0
                                    sb6=0
                                    sb7=0
                                    sbq1=0
                                    sbq2=0
                                    sbq3=0
                                    sbq4=0
                                    sbq5=0
                                    sbq6=0
                                    sbq7=0                        
                                    countp=0
                                    countpgq=0
                                    for yo in np.arange(minyoffset,maxyoffset+1,1):
                                        for xo in np.arange(minxoffset,maxxoffset+1,1):
                                            lonpi=(xo+0.5)*pixelWidth+originX
                                            latpi=(yo+0.5)*pixelHeight+originY
                                            [wlonpi,wlatpi]=pyproj.transform(sinu,wcg84,lonpi, latpi)
                                            if footprint.contains_point((wlonpi,wlatpi)):
                                                if (b1[yo,xo]>=0) & (b1[yo,xo]<=10000) & (b2[yo,xo]>=0) & (b2[yo,xo]<=10000) & (b3[yo,xo]>=0) & (b3[yo,xo]<=10000) & \
                                                (b4[yo,xo]>=0) & (b4[yo,xo]<=10000) & (b5[yo,xo]>=0) & (b5[yo,xo]<=10000) & (b6[yo,xo]>=0) & (b6[yo,xo]<=10000) & (b7[yo,xo]>=0) & (b7[yo,xo]<=10000):
                                                    
                                                    q=quality[yo,xo]
                                                    sb1=sb1+b1[yo,xo]
                                                    sb2=sb2+b2[yo,xo] 
                                                    sb3=sb3+b3[yo,xo]
                                                    sb4=sb4+b4[yo,xo]
                                                    sb5=sb5+b5[yo,xo]
                                                    sb6=sb6+b6[yo,xo]
                                                    sb7=sb7+b7[yo,xo]
                                                    
                                                    countp=countp+1
    
                                                    if q<1:
                                                        sbq1=sbq1+b1[yo,xo]
                                                        sbq2=sbq2+b2[yo,xo] 
                                                        sbq3=sbq3+b3[yo,xo]
                                                        sbq4=sbq4+b4[yo,xo]
                                                        sbq5=sbq5+b5[yo,xo]
                                                        sbq6=sbq6+b6[yo,xo]
                                                        sbq7=sbq7+b7[yo,xo]
                                                        
                                                        countpgq=countpgq+1
            
                                    if countp !=0:
                                        
                                        vc[nid,0]=sb1/countp
                                        vc[nid,1]=sb2/countp   
                                        vc[nid,2]=sb3/countp
                                        vc[nid,3]=sb4/countp
                                        vc[nid,4]=sb5/countp
                                        vc[nid,5]=sb6/countp
                                        vc[nid,6]=sb7/countp
                                        
                                        if countpgq !=0:                      
                                            vc[nid,7]=sbq1/countpgq
                                            vc[nid,8]=sbq2/countpgq  
                                            vc[nid,9]=sbq3/countpgq
                                            vc[nid,10]=sbq4/countpgq
                                            vc[nid,11]=sbq5/countpgq
                                            vc[nid,12]=sbq6/countpgq
                                            vc[nid,13]=sbq7/countpgq
                                i_break=i_break+1
                            if i_break !=0:
                                break            
                    
                        nid=nid+1                            
                    
                    nn=np.count_nonzero(vc[:,0])
                    if mn in [2,3,4]:
                        
                        if (nn>0):
                            dataset_all[i2,0]=(np.sum(vc[:,0]))/nn
                            dataset_all[i2,1]=(np.sum(vc[:,1]))/nn                    
                            dataset_all[i2,2]=(np.sum(vc[:,2]))/nn                   
                            dataset_all[i2,3]=(np.sum(vc[:,3]))/nn                    
                            dataset_all[i2,4]=(np.sum(vc[:,4]))/nn                    
                            dataset_all[i2,5]=(np.sum(vc[:,5]))/nn                   
                            dataset_all[i2,6]=(np.sum(vc[:,6]))/nn                   
                          
                            dataset_all[i2,7]=(np.sum(vc[:,7]))/nn
                            dataset_all[i2,8]=(np.sum(vc[:,8]))/nn                    
                            dataset_all[i2,9]=(np.sum(vc[:,9]))/nn                  
                            dataset_all[i2,10]=(np.sum(vc[:,10]))/nn                    
                            dataset_all[i2,11]=(np.sum(vc[:,11]))/nn                   
                            dataset_all[i2,12]=(np.sum(vc[:,12]))/nn                   
                            dataset_all[i2,13]=(np.sum(vc[:,13]))/nn
                                                   
                            floni=flons[ia,:]
                            flati=flats[ia,:]
                            
                            lon0=floni[0]
                            lon1=floni[1]
                            lon2=floni[2]
                            lon3=floni[3]
                            
                            lat0=flati[0]
                            lat1=flati[1]
                            lat2=flati[2]
                            lat3=flati[3]
                                                   
                            loni=lons[ia]
                            lati=lats[ia]
                        
                            lc_oco=lcsif[ia]
                            date_v=date[ia]
                            day_v=day[ia]
                            sif757_v=sif757[ia]
                            dcsif757_v=dcsif757[ia]
                            COSsif757_v=COSsif757[ia]
                            Asif757_v=Asif757[ia]
                            Adcsif757_v=Adcsif757[ia]
                            ACOSsif757_v=ACOSsif757[ia]    
                            sif771_v=sif771[ia]
                            dcsif771_v=dcsif771[ia]
                            COSsif771_v=COSsif771[ia]
                            Asif771_v=Asif771[ia]
                            Adcsif771_v=Adcsif771[ia]
                            ACOSsif771_v=ACOSsif771[ia]     
                                  
                            dataset_all[i2,14]=sif757_v
                            dataset_all[i2,15]=dcsif757_v
                            dataset_all[i2,16]=COSsif757_v
                            dataset_all[i2,17]=Asif757_v
                            dataset_all[i2,18]=Adcsif757_v
                            dataset_all[i2,19]=ACOSsif757_v
                            
                            dataset_all[i2,20]=sif771_v
                            dataset_all[i2,21]=dcsif771_v
                            dataset_all[i2,22]=COSsif771_v
                            dataset_all[i2,23]=Asif771_v
                            dataset_all[i2,24]=Adcsif771_v
                            dataset_all[i2,25]=ACOSsif771_v
                            
                            dataset_all[i2,26]=date_v
        
                            dataset_all[i2,27]=lc_oco
    
                                                      
                            dataset_all[i2,28]=loni
                            dataset_all[i2,29]=lati
                                                     
                            if (i2%100)==0:
                                np.save('dataset'+str(mn)+'.npy',dataset_all)
                            i2=i2+1
                            print mn,i2,ia,train_num,LC_num
                            
                    else:
                        if (nn==5):
                            
                            dataset_all[i2,0]=np.mean(vc[:,0])
                            dataset_all[i2,1]=np.mean(vc[:,1])                    
                            dataset_all[i2,2]=np.mean(vc[:,2])                    
                            dataset_all[i2,3]=np.mean(vc[:,3])                    
                            dataset_all[i2,4]=np.mean(vc[:,4])                    
                            dataset_all[i2,5]=np.mean(vc[:,5])                    
                            dataset_all[i2,6]=np.mean(vc[:,6])                    
                          
                            dataset_all[i2,7]=np.mean(vc[:,7])
                            dataset_all[i2,8]=np.mean(vc[:,8])                    
                            dataset_all[i2,9]=np.mean(vc[:,9])                    
                            dataset_all[i2,10]=np.mean(vc[:,10])                    
                            dataset_all[i2,11]=np.mean(vc[:,11])                    
                            dataset_all[i2,12]=np.mean(vc[:,12])                    
                            dataset_all[i2,13]=np.mean(vc[:,13])
                                                   
                            floni=flons[ia,:]
                            flati=flats[ia,:]
                            
                            lon0=floni[0]
                            lon1=floni[1]
                            lon2=floni[2]
                            lon3=floni[3]
                            
                            lat0=flati[0]
                            lat1=flati[1]
                            lat2=flati[2]
                            lat3=flati[3]
                                                   
                            loni=lons[ia]
                            lati=lats[ia]
                        
                            lc_oco=lcsif[ia]
                            date_v=date[ia]
                            day_v=day[ia]
                            sif757_v=sif757[ia]
                            dcsif757_v=dcsif757[ia]
                            COSsif757_v=COSsif757[ia]
                            Asif757_v=Asif757[ia]
                            Adcsif757_v=Adcsif757[ia]
                            ACOSsif757_v=ACOSsif757[ia]    
                            sif771_v=sif771[ia]
                            dcsif771_v=dcsif771[ia]
                            COSsif771_v=COSsif771[ia]
                            Asif771_v=Asif771[ia]
                            Adcsif771_v=Adcsif771[ia]
                            ACOSsif771_v=ACOSsif771[ia]     
                                  
                            dataset_all[i2,14]=sif757_v
                            dataset_all[i2,15]=dcsif757_v
                            dataset_all[i2,16]=COSsif757_v
                            dataset_all[i2,17]=Asif757_v
                            dataset_all[i2,18]=Adcsif757_v
                            dataset_all[i2,19]=ACOSsif757_v
                            
                            dataset_all[i2,20]=sif771_v
                            dataset_all[i2,21]=dcsif771_v
                            dataset_all[i2,22]=COSsif771_v
                            dataset_all[i2,23]=Asif771_v
                            dataset_all[i2,24]=Adcsif771_v
                            dataset_all[i2,25]=ACOSsif771_v
                            
                            dataset_all[i2,26]=date_v
        
                            dataset_all[i2,27]=lc_oco
    
                                                      
                            dataset_all[i2,28]=loni
                            dataset_all[i2,29]=lati
                                                     
                            if (i2%100)==0:
                                np.save('dataset'+str(mn)+'.npy',dataset_all)
                            i2=i2+1
                            print mn,i2,ia,train_num,LC_num

    dataset_all=dataset_all[:i2,:]
    np.save('/local/workdir/Yu/workdir1015/201508a/data/dataset'+str(mn)+'.npy',dataset_all)
################################################################################################################
#training
names = locals()
sif_column=19
mn=19
n1l1=np.arange(3,17,2)
n2l1=np.arange(3,17,2)
n2l2=np.arange(3,17,2)
n3l1 =np.arange(3,17,2)
n3l2=np.arange(3,17,2)
n3l3=np.arange(3,17,2)
def rmse(y_pred,y_test):
    sum_mean=0  
    for i in range(len(y_pred)):  
        sum_mean+=(y_pred[i]-y_test[i])**2  
    sum_erro=np.sqrt(sum_mean/(len(y_test)))  
    return sum_erro  
   
def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2
performance=np.zeros((mn,6))
statistics=np.zeros((mn,4))
for m in np.arange(mn):
    data0=np.load('/local/workdir/Yu/workdir1015/201508a/data/dataset'+str(m)+'.npy')
    print 'clearing'
    idx_zeros=[]
    idx=np.arange(len(data0))
    for i in idx:
        if (np.count_nonzero(data0[i,:7])==0):
            idx_zeros.append(i)           
    data0=np.delete(data0, idx_zeros, 0)  
        
    sif2=(data0[:,18]+data0[:,24]*1.5)/2
    data0[:,19]=sif2
    x=1
    idx_abn=[]
    idx1=np.arange(len(data0))
    ncol=data0.shape[1]
    for i1 in idx1:
        for i2 in [0,1,2,3,4,5,6]:
            if (data0[i1,i2]<=10000) & (data0[i1,i2]>=0):
                x=1
            else:
                idx_abn.append(i1)
        for i3 in [18,19]:
            if (data0[i1,i3]<=8) & (data0[i1,i3]>-5):
                x=1
            else:
                idx_abn.append(i1)
    print len(idx_abn)
    data0=np.delete(data0, idx_abn, 0)
    
    idx_random=np.random.permutation(data0.shape[0])
    data0=data0[idx_random]
    data=data0
    datanum=len(data) 
    statistics[m,0]=m
    if datanum > 0:
        sif=data[:,sif_column]
        maxs=sif.max()
        mins=sif.min()
        means=sif.mean()
        statistics[m,1]=maxs
        statistics[m,2]=mins
        statistics[m,3]=means        
        np.save('/local/workdir/Yu/workdir1015/201508a/train/statistics.npy',statistics) 
                   
    if datanum>10:
        performance[m,0]=m
        
        data_feature=data[:,:7]
        data_label=data[:,sif_column]
        scaler = StandardScaler()
        scaler.fit(data_feature)  
        with open('/local/workdir/Yu/workdir1015/201508a/train/scaler'+str(m)+'.pickle','wb') as f:
            pickle.dump(scaler,f)
    ###########################################################
        num_p=int(0.2*datanum)
 
        data_p1=data[:num_p+1,:]
        data_p2=data[num_p+1:(2*num_p+1),:]
        data_p3=data[(2*num_p+1):(3*num_p+1),:]
        data_p4=data[(3*num_p+1):(4*num_p+1),:]
        data_p5=data[(4*num_p+1):,:]
        
        data1=np.concatenate([data_p1,data_p2,data_p3,data_p4])
        data2=data_p5
        X11=data1[:,:7]
        Y11=data1[:,sif_column]
        X12=data2[:,:7]
        Y12=data2[:,sif_column]
        
        data1=np.concatenate([data_p1,data_p2,data_p3,data_p5])
        data2=data_p4
        X21=data1[:,:7]
        Y21=data1[:,sif_column]
        X22=data2[:,:7]
        Y22=data2[:,sif_column]
        
        data1=np.concatenate([data_p1,data_p2,data_p4,data_p5])
        data2=data_p3
        X31=data1[:,:7]
        Y31=data1[:,sif_column]
        X32=data2[:,:7]
        Y32=data2[:,sif_column]
        
        data1=np.concatenate([data_p1,data_p3,data_p4,data_p5])
        data2=data_p2
        X41=data1[:,:7]
        Y41=data1[:,sif_column]
        X42=data2[:,:7]
        Y42=data2[:,sif_column]
        
        data1=np.concatenate([data_p2,data_p3,data_p4,data_p5])
        data2=data_p1
        X51=data1[:,:7]
        Y51=data1[:,sif_column]
        X52=data2[:,:7]
        Y52=data2[:,sif_column]         
  
        minrmse=100
        for n1 in n1l1:
            print 'modle',m,' ann',n1
            r2_ann_sum=0
            rmse_ann_sum=0
            rmse_ann_sum=0
            for i in [1,2,3,4,5]:

                X_1=(names['X%s' % (i*10+1)])
                Y_1=(names['Y%s' % (i*10+1)])
                X_2=(names['X%s' % (i*10+2)])
                Y_2=(names['Y%s' % (i*10+2)])
            
                X1 = scaler.transform(X_1)
                Y1=Y_1
                X2 = scaler.transform(X_2)
                Y2=Y_2
            
                yt=Y2
                
                ann = MLPRegressor( hidden_layer_sizes=(n1),solver='lbfgs', alpha=1e-5,random_state=1,max_iter=10000)
                ann.fit(X1, Y1)
                y2_ann=ann.predict(X2)
                y1_ann=ann.predict(X1)
                yp = y2_ann

                r2v_ann=rsquared(yp,yt)
                r2_ann_sum=r2_ann_sum+r2v_ann
                
                rmse_ann=rmse(yp,yt)
                rmse_ann_sum = rmse_ann_sum+rmse_ann
        
                names['tt'+str(i)]=Y1
                names['tp'+str(i)]=y1_ann
                names['vt'+str(i)]=yt
                names['vp'+str(i)]=yp
        
            r2_ann_mean=r2_ann_sum/5
            rmse_ann_mean=rmse_ann_sum/5
            
            if rmse_ann_mean < minrmse:
                minrmse=rmse_ann_mean
                best_n1=n1
                best_n2=0
                best_n3=0
                r2=r2_ann_mean
                
                X1=scaler.transform(data_feature)
                Y1=data_label
                ann = MLPRegressor( hidden_layer_sizes=(best_n1,),solver='lbfgs', alpha=1e-5,random_state=1,max_iter=10000)
                ann.fit(X1, Y1)
                with open('/local/workdir/Yu/workdir1015/201508a/train/ann'+str(m)+'.pickle','wb') as f:
                    pickle.dump(ann,f)
                    
                tt=np.concatenate([tt1,tt2,tt3,tt4,tt5])
                tp=np.concatenate([tp1,tp2,tp3,tp4,tp5])    
                vt=np.concatenate([vt1,vt2,vt3,vt4,vt5])    
                vp=np.concatenate([vp1,vp2,vp3,vp4,vp5])
                
                np.save('/local/workdir/Yu/workdir1015/201508a/train/yt_train_m'+str(m)+'.npy',tt)
                np.save('/local/workdir/Yu/workdir1015/201508a/train/yp_train_m'+str(m)+'.npy',tp)
                np.save('/local/workdir/Yu/workdir1015/201508a/train/yt_validation_m'+str(m)+'.npy',vt)
                np.save('/local/workdir/Yu/workdir1015/201508a/train/yp_validation_m'+str(m)+'.npy',vp)   
            
                    
        for n1 in n2l1:
            for n2 in n2l2:

                print 'modle',m,' ann',n1,n2
                r2_ann_sum=0
                rmse_ann_sum=0
                rmse_ann_sum=0
                for i in [1,2,3,4,5]:

                    X_1=(names['X%s' % (i*10+1)])
                    Y_1=(names['Y%s' % (i*10+1)])
                    X_2=(names['X%s' % (i*10+2)])
                    Y_2=(names['Y%s' % (i*10+2)])
                
                    X1 = scaler.transform(X_1)
                    Y1=Y_1
                    X2 = scaler.transform(X_2)
                    Y2=Y_2
                
                    yt=Y2
                    
                    ann = MLPRegressor( hidden_layer_sizes=(n1,n2),solver='lbfgs', alpha=1e-5,random_state=1,max_iter=10000)
                    ann.fit(X1, Y1)
                    y2_ann=ann.predict(X2)
                    y1_ann=ann.predict(X1)
                    yp = y2_ann
                    
                    
                    r2v_ann=rsquared(yp,yt)
                    r2_ann_sum=r2_ann_sum+r2v_ann
                    
                    rmse_ann=rmse(yp,yt)
                    rmse_ann_sum = rmse_ann_sum+rmse_ann
            
                    names['tt'+str(i)]=Y1
                    names['tp'+str(i)]=y1_ann
                    names['vt'+str(i)]=yt
                    names['vp'+str(i)]=yp
            
                r2_ann_mean=r2_ann_sum/5
                rmse_ann_mean=rmse_ann_sum/5
                
                if rmse_ann_mean < minrmse:
                    minrmse=rmse_ann_mean
                    best_n1=n1
                    best_n2=n2
                    best_n3=0
                    r2=r2_ann_mean
                    
                    X1=scaler.transform(data_feature)
                    Y1=data_label
                    ann = MLPRegressor( hidden_layer_sizes=(best_n1,best_n2),solver='lbfgs', alpha=1e-5,random_state=1,max_iter=10000)
                    ann.fit(X1, Y1)
                    with open('/local/workdir/Yu/workdir1015/201508a/train/ann'+str(m)+'.pickle','wb') as f:
                        pickle.dump(ann,f)
                        
                    tt=np.concatenate([tt1,tt2,tt3,tt4,tt5])
                    tp=np.concatenate([tp1,tp2,tp3,tp4,tp5])    
                    vt=np.concatenate([vt1,vt2,vt3,vt4,vt5])    
                    vp=np.concatenate([vp1,vp2,vp3,vp4,vp5])
                    
                    np.save('/local/workdir/Yu/workdir1015/201508a/train/yt_train_m'+str(m)+'.npy',tt)
                    np.save('/local/workdir/Yu/workdir1015/201508a/train/yp_train_m'+str(m)+'.npy',tp)
                    np.save('/local/workdir/Yu/workdir1015/201508a/train/yt_validation_m'+str(m)+'.npy',vt)
                    np.save('/local/workdir/Yu/workdir1015/201508a/train/yp_validation_m'+str(m)+'.npy',vp)   
                            
        for n1 in n3l1:
            for n2 in n3l2:
                for n3 in n3l3:
                    print 'modle',m,' ann',n1,n2,n3
                    r2_ann_sum=0
                    rmse_ann_sum=0
                    rmse_ann_sum=0
                    for i in [1,2,3,4,5]:

                        X_1=(names['X%s' % (i*10+1)])
                        Y_1=(names['Y%s' % (i*10+1)])
                        X_2=(names['X%s' % (i*10+2)])
                        Y_2=(names['Y%s' % (i*10+2)])
                    
                        X1 = scaler.transform(X_1)
                        Y1=Y_1
                        X2 = scaler.transform(X_2)
                        Y2=Y_2
                    
                        yt=Y2
                        
                        ann = MLPRegressor( hidden_layer_sizes=(n1,n2,n3),solver='lbfgs', alpha=1e-5,random_state=1,max_iter=10000)
                        ann.fit(X1, Y1)
                        y2_ann=ann.predict(X2)
                        y1_ann=ann.predict(X1)
                        yp = y2_ann

                        r2v_ann=rsquared(yp,yt)
                        r2_ann_sum=r2_ann_sum+r2v_ann
                        
                        rmse_ann=rmse(yp,yt)
                        rmse_ann_sum = rmse_ann_sum+rmse_ann
                        
                        names['tt'+str(i)]=Y1
                        names['tp'+str(i)]=y1_ann
                        names['vt'+str(i)]=yt
                        names['vp'+str(i)]=yp
                
                    r2_ann_mean=r2_ann_sum/5
                    rmse_ann_mean=rmse_ann_sum/5
                    
                    if rmse_ann_mean < minrmse:
                        minrmse=rmse_ann_mean
                        best_n1=n1
                        best_n2=n2
                        best_n3=n3
                        r2=r2_ann_mean
                        
                        X1=scaler.transform(data_feature)
                        Y1=data_label
                        ann = MLPRegressor( hidden_layer_sizes=(best_n1,best_n2,best_n3),solver='lbfgs', alpha=1e-5,random_state=1,max_iter=10000)
                        ann.fit(X1, Y1)
                        with open('/local/workdir/Yu/workdir1015/201508a/train/ann'+str(m)+'.pickle','wb') as f:
                            pickle.dump(ann,f)
                            
                        tt=np.concatenate([tt1,tt2,tt3,tt4,tt5])
                        tp=np.concatenate([tp1,tp2,tp3,tp4,tp5])    
                        vt=np.concatenate([vt1,vt2,vt3,vt4,vt5])    
                        vp=np.concatenate([vp1,vp2,vp3,vp4,vp5])
                        
                        np.save('/local/workdir/Yu/workdir1015/201508a/train/yt_train_m'+str(m)+'.npy',tt)
                        np.save('/local/workdir/Yu/workdir1015/201508a/train/yp_train_m'+str(m)+'.npy',tp)
                        np.save('/local/workdir/Yu/workdir1015/201508a/train/yt_validation_m'+str(m)+'.npy',vt)
                        np.save('/local/workdir/Yu/workdir1015/201508a/train/yp_validation_m'+str(m)+'.npy',vp)   
                        
        performance[m,1]=best_n1
        performance[m,2]=best_n2
        performance[m,3]=best_n3
        
        performance[m,4]=minrmse
        performance[m,5]=r2
    
        np.save('/local/workdir/Yu/workdir1015/201508a/train/performance.npy',performance)        
#######################################################################################################################################    
#prediction
mlc=np.load('/local/workdir/Yu/mlc2012.npy')

print 'preparing global predictors'

b1 = np.loadtxt('/local1/storage/data/BRDF005/e4ftl01.cr.usgs.gov/MOTA/MCD43C4.006/16day/MCD43C4band1201508a.txt')
print 'preditors 1 processing'
b2 = np.loadtxt('/local1/storage/data/BRDF005/e4ftl01.cr.usgs.gov/MOTA/MCD43C4.006/16day/MCD43C4band2201508a.txt')
print 'preditors 2 processing'
b3 = np.loadtxt('/local1/storage/data/BRDF005/e4ftl01.cr.usgs.gov/MOTA/MCD43C4.006/16day/MCD43C4band3201508a.txt')
print 'preditors 3 processing'
b4 = np.loadtxt('/local1/storage/data/BRDF005/e4ftl01.cr.usgs.gov/MOTA/MCD43C4.006/16day/MCD43C4band4201508a.txt')
print 'preditors 4 processing'
b5 = np.loadtxt('/local1/storage/data/BRDF005/e4ftl01.cr.usgs.gov/MOTA/MCD43C4.006/16day/MCD43C4band5201508a.txt')
print 'preditors 5 processing'
b6 = np.loadtxt('/local1/storage/data/BRDF005/e4ftl01.cr.usgs.gov/MOTA/MCD43C4.006/16day/MCD43C4band6201508a.txt')
print 'preditors 6 processing'
b7 = np.loadtxt('/local1/storage/data/BRDF005/e4ftl01.cr.usgs.gov/MOTA/MCD43C4.006/16day/MCD43C4band7201508a.txt')
print 'preditors 7 processing'
#
mb1=b1.reshape((3600,7200))
mb2=b2.reshape((3600,7200))
mb3=b3.reshape((3600,7200))
mb4=b4.reshape((3600,7200))
mb5=b5.reshape((3600,7200))
mb6=b6.reshape((3600,7200))
mb7=b7.reshape((3600,7200))

mb1=mb1[::-1]
mb2=mb2[::-1]
mb3=mb3[::-1]
mb4=mb4[::-1]
mb5=mb5[::-1]
mb6=mb6[::-1]
mb7=mb7[::-1]

nlats=3600
nlons=7200

flt=np.zeros((nlats,nlons))#feature
idx1=np.arange(nlats)
idx2=np.arange(nlons)

for i1 in idx1:   
    print i1
    for i2 in idx2:
        if (i1<600):
            flt[i1,i2]=-1
          
        elif mb1[i1,i2]==-1:
            flt[i1,i2]=-1

        elif mb2[i1,i2]==-1:
            flt[i1,i2]=-1

        elif (mb3[i1,i2]==-1):
            flt[i1,i2]=-1

        elif (mb4[i1,i2]==-1):
            flt[i1,i2]=-1
   
        elif (mb5[i1,i2]==-1):
            flt[i1,i2]=-1
       
        elif (mb6[i1,i2]==-1):
            flt[i1,i2]=-1
      
        elif (mb7[i1,i2]==-1):
            flt[i1,i2]=-1

np.save('/local/workdir/Yu/workdir1015/201508a/predict/predictor_mask.npy',flt)

totalnum=1500*nlons

pf=np.zeros((totalnum,10))#feature

idx1=np.arange(nlats)
idx11=idx1[600:]#exclude Antarctica
idx2=np.arange(nlons)
count=0
i3=0
lcrange=[1,2,3,4,5,6,7,8,9,10,12,14]#IGBP index
for i1 in idx11:
    print i1,i3    
    for i2 in idx2:
        count=count+1
        lc_mod=mlc[i1,i2]  
        if lc_mod in lcrange:
        
            pf[i3,0]=mb1[i1,i2]
            pf[i3,1]=mb2[i1,i2]    
            pf[i3,2]=mb3[i1,i2]
            pf[i3,3]=mb4[i1,i2]
            pf[i3,4]=mb5[i1,i2]
            pf[i3,5]=mb6[i1,i2]
            pf[i3,6]=mb7[i1,i2]
            
            if (lc_mod==1) or (lc_mod==3):
                if i1>1800:
                    pf[i3,7]=0
                if i1<=1800:
                    pf[i3,7]=1
            if  (lc_mod==2):
                if i2<3100:
                    pf[i3,7]=2
                if (i2>=3100)&(i2<4800):
                    pf[i3,7]=3   
                if i2>=4800:
                    pf[i3,7]=4    
            if (lc_mod==4) or (lc_mod==5):
                if i1>1800:
                    pf[i3,7]=5
                if i1<=1800:
                    pf[i3,7]=6                    
            if (lc_mod==6) or (lc_mod==7):
                if i1>1800:
                    pf[i3,7]=7
                if i1<=1800:
                    pf[i3,7]=8                    
            if (lc_mod==8) or (lc_mod==9):
                if i1>1800:
                    pf[i3,7]=9
                if i1<=1800:
                    pf[i3,7]=10                    
            if (lc_mod==10):
                if i1>1800:
                    pf[i3,7]=11
                if i1<=1800:
                    pf[i3,7]=12   
            if (lc_mod==12) or (lc_mod==14):
                if ((i1>1800) & (i2<3100)):
                    pf[i3,7]=13
                if ((i1<=1800) & (i2<3100)):
                    pf[i3,7]=14                       
                if ((i1>1800) & (i2>=3100) & (i2<4800)):
                    pf[i3,7]=15
                if ((i1<=1800) & (i2>=3100) & (i2<4800)):
                    pf[i3,7]=16 
                if ((i1>1800) & (i2>=4800)):
                    pf[i3,7]=17
                if ((i1<=1800) & (i2>=4800)):
                    pf[i3,7]=18                     
            pf[i3,8]=i1
            pf[i3,9]=i2
                        
            i3=i3+1
np.save('/local/workdir/Yu/workdir1015/201508a/predict/predict_feature.npy',pf)
#######################################################################################################################
names = locals()
pf=np.load('/local/workdir/Yu/workdir1015/201508a/predict/predict_feature.npy')
print 'predictors loaded'

lcrange=np.array([1,2,3,4,5,6,7,8,9,10,11,12])

for m in np.arange(19):
        
    pf_m=pf[pf[:,7]==m,:]
    np.save('pf_'+str(m)+'.npy',pf_m)
print 'model devided'

print 'predicting'
#sif_rf=np.zeros((3600,7200))
sif_ann=np.zeros((3600,7200))

for m in np.arange(19):

        
    print 'model', m
     
    pf0=np.load('pf_'+str(m)+'.npy')
    dl=len(pf0) 
    print dl
    if dl>0:
        
        feature=pf0[:,:7]
      
        with open('/local/workdir/Yu/workdir1015/201508a/train/scaler'+str(m)+'.pickle','rb') as f:
            scaler=pickle.load(f)
         
        with open('/local/workdir/Yu/workdir1015/201508a/train/ann'+str(m)+'.pickle','rb') as f:
            annm=pickle.load(f)
                
        f0 = scaler.transform(feature)
    
        print 'scaled'

        pv_ann=annm.predict(f0)
        print 'ann finished'
    
        num=len(pf0)
        
        for i in np.arange(num):
    
            i1=int(pf0[i,8])
            i2=int(pf0[i,9])

            sif_ann[i1,i2]=pv_ann[i]

np.save('/local/workdir/Yu/workdir1015/201508a/predict/OCO2_RSIF_ann.npy',sif_ann)
    
################################################################################################to nc
#################################################################################################
#lon limits
l1=-180
l2=180
#lat limits
r1=-90
r2=90

nlats=3600
nlons=7200
#nlats=1200
#nlons=2390
lats=np.arange(r1+0.025,r2,0.05)
lons=np.arange(l1+0.025,l2,0.05)

#lon0, lat0=np.meshgrid(lons,lats)
    
flt=np.load('/local/workdir/Yu/workdir1015/201508a/predict/predictor_mask.npy')
#
mask=(flt<0)
sif_ann_masked=ma.array(sif_ann,mask=mask)

f = nc4.Dataset('/local/workdir/Yu/workdir1015/201508a/predict/sif_ann.nc4','w', format='NETCDF4') #'w' stands for write
nx = 3600; ny = 7200

f.createDimension('x',nx)
f.createDimension('y',ny)

longitude = f.createVariable('longitude', 'float64', 'y')
latitude = f.createVariable('latitude', 'float64', 'x')  
SIF= f.createVariable('sif_ann', 'float64', ('x','y'))

longitude[:] = lons #The "[:]" at the end of the variable instance is necessary
latitude[:] = lats
SIF[:] = sif_ann_masked
#f.description = "This dataset containing lon, lat, sif_ann"
f.close()   
