import shutil
import numpy as np
import time as T
from joblib import Parallel, delayed
from scipy.interpolate import griddata,interpn,RegularGridInterpolator
import math
import os
import cv2
from ftplib import FTP
import rioxarray
import xarray as xr
import paramiko
from scp import SCPClient
import subprocess
import datetime as dt
import numba as nb
import concurrent.futures

SZA_PATH = '/data01/GEO/INPUT/ANGLE/Solar_Zenith_Angle_u2/'
SAA_PATH = '/data01/GEO/INPUT/ANGLE/Solar_Azimuth_Angle_u2/'

LUT_PATH = '/data01/GEO/INPUT/LUT/'
CAMS_PATH = '/data01/GEO/INPUT/ATMOSPHERE/'
DN_PATH = '/data01/GEO/INPUT/'
CAMS_AERO_PATH = '/data01/GEO/INPUT/AEROSOL_TYPE/'


sza = np.linspace(0,80,17)
vza = np.linspace(0,80,17)
water = np.linspace(0,7,8)
ozone = np.linspace(0.2,0.4,5)
AOT = np.array([0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0,1.5,2.0])
raa = np.linspace(0,180,19)
al = np.linspace(0,8,5)
aero_type = np.array([0,1])

def read_AHI_AL(res):
    paths = {
        0.005: ('/data01/GEO/INPUT/ELEVATION_GEO/AHI/MERIT_DEM_AHI_05km.dat', 24000),
        0.01: ('/data01/GEO/INPUT/ELEVATION_GEO/AHI/MERIT_DEM_AHI_10km.dat', 12000),
        0.02: ('/data01/GEO/INPUT/ELEVATION_GEO/AHI/MERIT_DEM_AHI_20km.dat', 6000)
    }
    AL_PATH, col = paths.get(res, (None, None))
    if AL_PATH:
        with open(AL_PATH,'rb') as fp:
            AL = np.frombuffer(fp.read(),dtype='u2').reshape(col,col) * 0.001
        return AL
    else:
        return None

def read_AHI_VA(res):
    paths = {
        0.005: ('/data01/GEO/INPUT/ANGLE/Viewer_Zenith_Angle/AHI_VZA_05.dat', '/data01/GEO/INPUT/ANGLE/Viewer_Azimuth_Angle/AHI_VAA_05.dat', 24000),
        0.01: ('/data01/GEO/INPUT/ANGLE/Viewer_Zenith_Angle/AHI_VZA_10.dat', '/data01/GEO/INPUT/ANGLE/Viewer_Azimuth_Angle/AHI_VAA_10.dat', 12000),
        0.02: ('/data01/GEO/INPUT/ANGLE/Viewer_Zenith_Angle/AHI_VZA_20.dat', '/data01/GEO/INPUT/ANGLE/Viewer_Azimuth_Angle/AHI_VAA_20.dat', 6000)
    }
    VZA_PATH, VAA_PATH, col = paths.get(res, (None, None, None))
    if VZA_PATH and VAA_PATH:
        with open(VZA_PATH,'rb') as fp:
            VZA = np.frombuffer(fp.read(),dtype='u2').reshape(col,col) / 100
        with open(VAA_PATH,'rb') as fp:
            VAA = np.frombuffer(fp.read(),dtype='u2').reshape(col,col) / 100
        return VZA,VAA
    else:
        return None
    
class H8_data:
    def __init__(self , account , pw , band , band_number , date , sava_path):
        self.account = account
        self.pw = pw
        self.band = band
        self.band_number = band_number
        self.date = date
        self.sava_path = sava_path

    def get_path(self):
        return '/data01/GEO/ORGDATA/H8AHI/hmwr829gr.cr.chiba-u.ac.jp/gridded/FD/V20151105/' + self.date[0:6] + '/' + self.band.upper() + '/'

    def get_filename(self):
        return self.date + "." + self.band + "." + self.band_number + ".fld.geoss.bz2"

    def DN2TBB(self, data):
        LUT=np.loadtxt(DN_PATH + 'count2tbb_v102/' + self.band + "." + self.band_number)
        return LUT[data,1]

    def file_path(self):
        return self.get_path() + self.get_filename()

    def download_H8data(self):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname='10.4.104.140', port=22, username=self.account, password=self.pw)
        scp = SCPClient(client.get_transport())
        sftp = client.open_sftp()

        try:
            sftp.stat(self.file_path())
        except FileNotFoundError:
            client.close()
            print("File Not Found")
            return 'No data'
        else:
            scp.get(self.file_path(), self.sava_path+'/')
            p = subprocess.Popen('lbzip2 -d {}{}'.format(self.sava_path+'/',self.file_path()[-33:]),shell=True)
            p.communicate()
            client.close()
            print ('Himawari8/AHI data Processed Finish')
            return self.sava_path + self.get_filename()[:-4]

    def read_H8data(self):
        H8_file_path = self.download_H8data()
        if self.band == "vis":
            sr = 12000
        elif self.band == "ext":
            sr = 24000
        else:
            sr = 6000
        if H8_file_path != 'No data':
            data = np.memmap(H8_file_path, dtype='>u2', mode='r', shape=(sr,sr))
            data = self.DN2TBB(data)
            data = data/100
            if self.band == "ext": 
                data = data.reshape(12000,2,12000,2).mean(-1).mean(1)
                
            print("data reading finish")
            return data
        else:
            return 'No data'
        

class LUT_interpolation:
    def __init__(self, band):
        self.band = band
        self.function_dict = {1: self._interpolation_band1,
                              2: self._interpolation_band2,
                              3: self._interpolation_band3,
                              4: self._interpolation_band4,
                              5: self._interpolation_band5,
                              6: self._interpolation_band6}
    
    def _interpolation_band1(self):
        X1 = np.loadtxt(LUT_PATH + "01_band1.csv",delimiter=",").reshape(2,5,12,5,17,17,19)
        X2 = np.loadtxt(LUT_PATH + "02_band1.csv",delimiter=",").reshape(2,5,12,5,17,17,19)
        X3 = np.loadtxt(LUT_PATH + "03_band1.csv",delimiter=",").reshape(2,5,12,5,17,17,19)
        fn1 = RegularGridInterpolator((aero_type,ozone,AOT,al,sza,vza,raa),X1,bounds_error=False,fill_value=np.nan)
        fn2 = RegularGridInterpolator((aero_type,ozone,AOT,al,sza,vza,raa),X2,bounds_error=False,fill_value=np.nan)
        fn3 = RegularGridInterpolator((aero_type,ozone,AOT,al,sza,vza,raa),X3,bounds_error=False,fill_value=np.nan)
        return fn1, fn2, fn3
    
    def _interpolation_band2(self):
        X1 = np.loadtxt(LUT_PATH + "01_band2.csv",delimiter=",").reshape(2,5,12,5,17,17,19)
        X2 = np.loadtxt(LUT_PATH + "02_band2.csv",delimiter=",").reshape(2,5,12,5,17,17,19)
        X3 = np.loadtxt(LUT_PATH + "03_band2.csv",delimiter=",").reshape(2,5,12,5,17,17,19)
        fn1 = RegularGridInterpolator((aero_type,ozone,AOT,al,sza,vza,raa),X1,bounds_error=False,fill_value=np.nan)
        fn2 = RegularGridInterpolator((aero_type,ozone,AOT,al,sza,vza,raa),X2,bounds_error=False,fill_value=np.nan)
        fn3 = RegularGridInterpolator((aero_type,ozone,AOT,al,sza,vza,raa),X3,bounds_error=False,fill_value=np.nan)
        return fn1, fn2, fn3
    
    def _interpolation_band3(self):
        X1 = np.loadtxt(LUT_PATH + "01_band3.csv",delimiter=",").reshape(2,8,5,12,5,17,17,19)
        X2 = np.loadtxt(LUT_PATH + "02_band3.csv",delimiter=",").reshape(2,8,5,12,5,17,17,19)
        X3 = np.loadtxt(LUT_PATH + "03_band3.csv",delimiter=",").reshape(2,8,5,12,5,17,17,19)        
        fn1 = RegularGridInterpolator((aero_type,water,ozone,AOT,al,sza,vza,raa),X1,bounds_error=False,fill_value=np.nan)
        fn2 = RegularGridInterpolator((aero_type,water,ozone,AOT,al,sza,vza,raa),X2,bounds_error=False,fill_value=np.nan)
        fn3 = RegularGridInterpolator((aero_type,water,ozone,AOT,al,sza,vza,raa),X3,bounds_error=False,fill_value=np.nan)
        return fn1, fn2, fn3
    
    def _interpolation_band4(self):
        X1 = np.loadtxt(LUT_PATH + "01_band4.csv",delimiter=",").reshape(2,8,12,5,17,17,19)
        X2 = np.loadtxt(LUT_PATH + "02_band4.csv",delimiter=",").reshape(2,8,12,5,17,17,19)
        X3 = np.loadtxt(LUT_PATH + "03_band4.csv",delimiter=",").reshape(2,8,12,5,17,17,19)
        fn1 = RegularGridInterpolator((aero_type,water,AOT,al,sza,vza,raa),X1,bounds_error=False,fill_value=np.nan)
        fn2 = RegularGridInterpolator((aero_type,water,AOT,al,sza,vza,raa),X2,bounds_error=False,fill_value=np.nan)
        fn3 = RegularGridInterpolator((aero_type,water,AOT,al,sza,vza,raa),X3,bounds_error=False,fill_value=np.nan)
        return fn1, fn2, fn3
    
    def _interpolation_band5(self):
        X1 = np.loadtxt(LUT_PATH + "01_band5.csv",delimiter=",").reshape(2,8,12,5,17,17,19)
        X2 = np.loadtxt(LUT_PATH + "02_band5.csv",delimiter=",").reshape(2,8,12,5,17,17,19)
        X3 = np.loadtxt(LUT_PATH + "03_band5.csv",delimiter=",").reshape(2,8,12,5,17,17,19)        
        fn1 = RegularGridInterpolator((aero_type,water,AOT,al,sza,vza,raa),X1,bounds_error=False,fill_value=np.nan)
        fn2 = RegularGridInterpolator((aero_type,water,AOT,al,sza,vza,raa),X2,bounds_error=False,fill_value=np.nan)
        fn3 = RegularGridInterpolator((aero_type,water,AOT,al,sza,vza,raa),X3,bounds_error=False,fill_value=np.nan)
        return fn1, fn2, fn3
    
    def _interpolation_band6(self):
        X1 = np.loadtxt(LUT_PATH + "01_band6.csv",delimiter=",").reshape(2,8,12,5,17,17,19)
        X2 = np.loadtxt(LUT_PATH + "02_band6.csv",delimiter=",").reshape(2,8,12,5,17,17,19)
        X3 = np.loadtxt(LUT_PATH + "03_band6.csv",delimiter=",").reshape(2,8,12,5,17,17,19)
        fn1 = RegularGridInterpolator((aero_type,water,AOT,al,sza,vza,raa),X1,bounds_error=False,fill_value=np.nan)
        fn2 = RegularGridInterpolator((aero_type,water,AOT,al,sza,vza,raa),X2,bounds_error=False,fill_value=np.nan)
        fn3 = RegularGridInterpolator((aero_type,water,AOT,al,sza,vza,raa),X3,bounds_error=False,fill_value=np.nan)
        return fn1, fn2, fn3

class AHI_angle:
    def __init__(self,date,col_AHI):
        # self.date = date[4:11]
        self.SZA_filename = SZA_PATH + 'AHI_SZA_2018{}5.dat'.format(date[4:11])
        self.SAA_filename = SAA_PATH + 'AHI_SAA_2018{}5.dat'.format(date[4:11])
        self.col_AHI = col_AHI
    def read_SolarAngle(self, filename):
        SA = np.memmap(filename, dtype='u2', mode='r', shape=(3000, 3000)) / 100
        SA=cv2.resize(np.array(SA,dtype='float64'),(self.col_AHI,self.col_AHI),interpolation=cv2.INTER_NEAREST)
        return SA
    
    def read_solar_angle(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            file_list = [self.SZA_filename, self.SAA_filename]
            results = [executor.submit(self.read_SolarAngle, file) for file in file_list]
        Result = [result.result() for result in results]
        return Result
    
    
class CAMS_data:
    def __init__(self,Y,M,D,H,MI,lat,lon,col_AHI):
        self.YYYY = Y
        self.MM = M
        self.DD = D
        self.HH = H
        self.MIN = MI
        self.lon = lon
        self.lat = lat
        self.col_AHI = col_AHI
        
    def read_CAMS(self):
        d1 = dt.datetime(int(self.YYYY),int(self.MM),int(self.DD))
        d2 = dt.datetime(int(self.YYYY),int(self.MM),int(self.DD)) + dt.timedelta(days=1)

        ds1 = xr.open_dataset(CAMS_PATH + d1.strftime('%Y') + d1.strftime('%m') + d1.strftime('%d') + '.nc')
        ds2 = xr.open_dataset(CAMS_PATH + d2.strftime('%Y') + d2.strftime('%m') + d2.strftime('%d') + '.nc')
        ds = xr.merge([ds1, ds2]) 
        ds = ds.interp(longitude=self.lon,latitude=self.lat,method="nearest")
        return ds
        
         
    # def CAMS_Temporal(self,ds):
    #     # dtime = dt.datetime(int(self.YYYY),int(self.MM),int(self.DD),int(self.HH),int(self.MIN)+5)
    #     e1= T.time()
    #     # ds = ds.interp(time = dtime,method = 'linear')
    #     e2 = T.time()
    #     print(e2-e1)
    #     OZ = ds['gtco3'].values        
    #     WV = ds['tcwv'].values        
    #     AOT550 = ds['aod550'].values
    #     WV = WV/10
    #     OZ = OZ*46.6975764
    #     OZ[OZ>=max(ozone)] = max(ozone)-(1/10000)
    #     OZ[OZ<=min(ozone)] = min(ozone)+(1/10000)
    #     WV[WV>=max(water)] = max(water)-(1/10000)
    #     WV[WV<=min(water)] = min(water)+(1/10000)
    #     AOT550[AOT550>=max(AOT)] = max(AOT)-(1/10000)
    #     AOT550[AOT550<=min(AOT)] = min(AOT)+(1/10000)
    #     return np.array(OZ).reshape(self.col_AHI,self.col_AHI),np.array(WV).reshape(self.col_AHI,self.col_AHI),np.array(AOT550).reshape(self.col_AHI,self.col_AHI)
    
    def CAMS_Temporal(self,ds):
        # dtime = dt.datetime(int(self.YYYY),int(self.MM),int(self.DD),int(self.HH),int(self.MIN)+5)
        # ds = ds.interp(time = dtime,method = 'linear')
        OZ = ds['gtco3'].values        
        WV = ds['tcwv'].values        
        AOT550 = ds['aod550'].values
        # e1= T.time()
        idx = int(self.HH) // 3
        past_hour = int(self.HH) % 3
        mint = past_hour * 60 + int(self.MIN)+5
        
        OZ = (((OZ[idx+1] - OZ[idx]) * mint / 180) + OZ[idx]) *46.6975764
        WV = (((WV[idx+1] - WV[idx]) * mint / 180) + WV[idx]) /10
        AOT550 = (((AOT550[idx+1] - AOT550[idx]) * mint / 180) + AOT550[idx])
        # e2 = T.time()
        # print(e2-e1)
 
        OZ = np.clip(OZ,0.2,0.4)
        WV = np.clip(WV,0,7)
        AOT550 = np.clip(AOT550,0,2)
        return OZ,WV,AOT550

    def read_CAMS_AERO(self):

        d1 = dt.datetime(int(self.YYYY),int(self.MM),int(self.DD))
        d2 = dt.datetime(int(self.YYYY),int(self.MM),int(self.DD)) + dt.timedelta(days=1)
        ds1 = xr.open_dataset(CAMS_AERO_PATH + d1.strftime('%Y') + d1.strftime('%m') + d1.strftime('%d') + '.nc')
        ds2 = xr.open_dataset(CAMS_AERO_PATH + d2.strftime('%Y') + d2.strftime('%m') + d2.strftime('%d') + '.nc')
        ds = xr.merge([ds1, ds2]) 
        # s = T.time()
        ds = ds.interp(longitude=self.lon,latitude=self.lat,method="nearest")
        # e1= T.time()
        # print(e1-s)
        return ds
    
    def CAMS_AERO_Temporal(self,ds):
        dtime = dt.datetime(int(self.YYYY),int(self.MM),int(self.DD),int(self.HH),int(self.MIN)+5)
        # e1= T.time()
        bc = ds['bcaod550'].values
        du = ds['duaod550'].values
        om = ds['omaod550'].values
        ss = ds['ssaod550'].values
        su = ds['suaod550'].values
        
        idx = int(self.HH) // 3
        past_hour = int(self.HH) % 3
        mint = past_hour * 60 + int(self.MIN)+5
        
        bc = (((bc[idx+1] - bc[idx]) * mint / 180) + bc[idx]) 
        du = (((du[idx+1] - du[idx]) * mint / 180) + du[idx])
        om = (((om[idx+1] - om[idx]) * mint / 180) + om[idx])
        ss = (((om[idx+1] - ss[idx]) * mint / 180) + ss[idx])
        su = (((om[idx+1] - su[idx]) * mint / 180) + su[idx])
        
        # e2 = T.time()
        # print(e2-e1)

        DL_6S = np.array(du)
        SL_6S = np.array(su) + np.array(bc)
        OC_6S = np.array(ss)
        WS_6S = np.array(om)

        Total = DL_6S + SL_6S + OC_6S + WS_6S
        precent_DL_6S = DL_6S / Total
        precent_SL_6S = SL_6S / Total
        precent_OC_6S = OC_6S / Total
        precent_WS_6S = WS_6S / Total
        P = np.dstack((precent_DL_6S,precent_WS_6S,precent_OC_6S,precent_SL_6S))
        Aerosol_type = np.where(np.amax(P,axis = 2) == precent_OC_6S,1,0)     
        return Aerosol_type
    
def H8_Process(ACCOUNT, PW, Band, Band_number, Date,Savepath):
    return H8_data(ACCOUNT, PW, Band, Band_number, Date,Savepath).read_H8data()

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def remove_original_file(path):
    shutil.rmtree(path)


def Time_split(time):
    date = time.strftime("%Y%m%d%H%M")
    YYYY, MM, DD, HH, MIN = date[0:4], date[4:6], date[6:8], date[8:10], date[10:12]
    return YYYY,MM,DD,HH,MIN,date

def read_Landmask(res):
    if res == 0.005:
        LAND_MASK_PATH = '/data01/GEO/INPUT/LAND_MASK/Landmask_05.dat'
        col = 24000
    elif res == 0.01:
        LAND_MASK_PATH = '/data01/GEO/INPUT/LAND_MASK/Landmask_10.dat' 
        col = 12000
    elif res == 0.02:
        LAND_MASK_PATH = '/data01/GEO/INPUT/LAND_MASK/Landmask_20.dat'
        col = 6000

    with open(LAND_MASK_PATH,'rb') as fp:
        LM = np.frombuffer(fp.read(),dtype='u1').reshape(col,col).astype('bool')
    return LM   


# Landmask = read_Landmask(res)

@nb.jit()
def get_water_idx(Landmask):
    Water_idx = []
    for i in range(12000):
        line_idx =[]
        for j in range(12000):
            if Landmask[i,j] == True:
                line_idx.append(j)

        Water_idx.append(line_idx)
    return Water_idx

def get_water_idx_2(Landmask):
    Water_idx = []
    for i in range(6000):
        line_idx =[]
        for j in range(6000):
            if Landmask[i,j] == True:
                line_idx.append(j)

        Water_idx.append(line_idx)
    return Water_idx

def calculate_6s_band1(fn1_1, fn2_1, fn3_1,Landmask,Aerosol_type,OZ,AOT550,RAA,AHI_SZA,AHI_VZA,AHI_AL,AHI_data,i):
    
    Aero_input = Aerosol_type[i][Landmask[i]]
    OZ_input = OZ[i][Landmask[i]]
    AOT550_input = AOT550[i][Landmask[i]]
    RAA_input = RAA[i][Landmask[i]]
    SZA_input = AHI_SZA[i][Landmask[i]]
    VZA_input = AHI_VZA[i][Landmask[i]]
    AL_input = AHI_AL[i][Landmask[i]]
    AHI_data_input = AHI_data[i][Landmask[i]]

    xi = np.array([Aero_input,OZ_input,AOT550_input,AL_input,SZA_input,VZA_input,RAA_input])
    xi = xi.T
    xa = fn1_1(xi)
    xb = fn2_1(xi)
    xc = fn3_1(xi)
    y = xa*AHI_data_input-xb
    SR = y/(1+xc*y)
    return SR

def calculate_6s_band2(fn1_2, fn2_2, fn3_2,Landmask,Aerosol_type,OZ,AOT550,RAA,AHI_SZA,AHI_VZA,AHI_AL,AHI_data,i):
    
    Aero_input = Aerosol_type[i][Landmask[i]]
    OZ_input = OZ[i][Landmask[i]]
    AOT550_input = AOT550[i][Landmask[i]]
    RAA_input = RAA[i][Landmask[i]]
    SZA_input = AHI_SZA[i][Landmask[i]]
    VZA_input = AHI_VZA[i][Landmask[i]]
    AL_input = AHI_AL[i][Landmask[i]]
    AHI_data_input = AHI_data[i][Landmask[i]]

    xi = np.array([Aero_input,OZ_input,AOT550_input,AL_input,SZA_input,VZA_input,RAA_input])
    xi = xi.T
    xa = fn1_2(xi)
    xb = fn2_2(xi)
    xc = fn3_2(xi)
    y = xa*AHI_data_input-xb
    SR = y/(1+xc*y)
    return SR


def calculate_6s_band3(fn1_3, fn2_3, fn3_3,Landmask,Aerosol_type,WV,OZ,AOT550,RAA,AHI_SZA,AHI_VZA,AHI_AL,AHI_data,i):
    
    Aero_input = Aerosol_type[i][Landmask[i]]
    WV_input = WV[i][Landmask[i]]
    OZ_input = OZ[i][Landmask[i]]
    AOT550_input = AOT550[i][Landmask[i]]
    RAA_input = RAA[i][Landmask[i]]
    SZA_input = AHI_SZA[i][Landmask[i]]
    VZA_input = AHI_VZA[i][Landmask[i]]
    AL_input = AHI_AL[i][Landmask[i]]
    AHI_data_input = AHI_data[i][Landmask[i]]

    xi = np.array([Aero_input,WV_input,OZ_input,AOT550_input,AL_input,SZA_input,VZA_input,RAA_input])
    xi = xi.T
    xa = fn1_3(xi)
    xb = fn2_3(xi)
    xc = fn3_3(xi)
    y = xa*AHI_data_input-xb
    SR = y/(1+xc*y)
    return SR 

def calculate_6s_band4(fn1_4, fn2_4, fn3_4,Landmask,Aerosol_type,WV,AOT550,RAA,AHI_SZA,AHI_VZA,AHI_AL,AHI_data,i):

    Aero_input = Aerosol_type[i][Landmask[i]]
    WV_input = WV[i][Landmask[i]]
    AOT550_input = AOT550[i][Landmask[i]]
    RAA_input = RAA[i][Landmask[i]]
    SZA_input = AHI_SZA[i][Landmask[i]]
    VZA_input = AHI_VZA[i][Landmask[i]]
    AL_input = AHI_AL[i][Landmask[i]]
    AHI_data_input = AHI_data[i][Landmask[i]]

    xi = np.array([Aero_input,WV_input,AOT550_input,AL_input,SZA_input,VZA_input,RAA_input])
    xi = xi.T
    xa = fn1_4(xi)
    xb = fn2_4(xi)
    xc = fn3_4(xi)
    y = xa*AHI_data_input-xb
    SR = y/(1+xc*y)
    return SR    

def calculate_6s_band5(fn1_4, fn2_4, fn3_4,Landmask,Aerosol_type,WV,AOT550,RAA,AHI_SZA,AHI_VZA,AHI_AL,AHI_data,i):

    Aero_input = Aerosol_type[i][Landmask[i]]
    WV_input = WV[i][Landmask[i]]
    AOT550_input = AOT550[i][Landmask[i]]
    RAA_input = RAA[i][Landmask[i]]
    SZA_input = AHI_SZA[i][Landmask[i]]
    VZA_input = AHI_VZA[i][Landmask[i]]
    AL_input = AHI_AL[i][Landmask[i]]
    AHI_data_input = AHI_data[i][Landmask[i]]

    xi = np.array([Aero_input,WV_input,AOT550_input,AL_input,SZA_input,VZA_input,RAA_input])
    xi = xi.T
    xa = fn1_4(xi)
    xb = fn2_4(xi)
    xc = fn3_4(xi)
    y = xa*AHI_data_input-xb
    SR = y/(1+xc*y)
    return SR  

def calculate_6s_band6(fn1_4, fn2_4, fn3_4,Landmask,Aerosol_type,WV,AOT550,RAA,AHI_SZA,AHI_VZA,AHI_AL,AHI_data,i):

    Aero_input = Aerosol_type[i][Landmask[i]]
    WV_input = WV[i][Landmask[i]]
    AOT550_input = AOT550[i][Landmask[i]]
    RAA_input = RAA[i][Landmask[i]]
    SZA_input = AHI_SZA[i][Landmask[i]]
    VZA_input = AHI_VZA[i][Landmask[i]]
    AL_input = AHI_AL[i][Landmask[i]]
    AHI_data_input = AHI_data[i][Landmask[i]]

    xi = np.array([Aero_input,WV_input,AOT550_input,AL_input,SZA_input,VZA_input,RAA_input])
    xi = xi.T
    xa = fn1_4(xi)
    xb = fn2_4(xi)
    xc = fn3_4(xi)
    y = xa*AHI_data_input-xb
    SR = y/(1+xc*y)
    return SR  