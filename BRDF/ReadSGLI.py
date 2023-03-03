import os
from ftplib import FTP
import rioxarray
import subprocess
import numpy as np
import datetime as dt
import ephem
import math

class FTP_Downloader:
    def __init__(self, ftp_addr, user_id, password):
        self.ftp_addr = ftp_addr
        self.user_id = user_id
        self.password = password
        
    def _login(self):
        self.ftp = FTP(self.ftp_addr)
        self.ftp.login(self.user_id, self.password)
    
    def _logout(self):
        self.ftp.quit()
        
    def download(self, remote_filepath, local_filepath):
        self._login()
        self.ftp.cwd(remote_filepath)
        list = self.ftp.nlst()
        if local_filepath in list:
            data = open(local_filepath, 'wb')
            self.ftp.retrbinary("RETR " + local_filepath, data.write, 1024)
            self._logout()
            return 'y'
        else:
            self._logout()
            return 'n'

class SGLI_L2_Downloader(FTP_Downloader):
    def __init__(self, User_ID, ProductName, Ver, Date, Tile, Obrit, ParaVer, Target_path):
        ftp_addr = 'ftp.gportal.jaxa.jp'
        super().init(ftp_addr, User_ID, 'anonymous')
        self.ProductName = ProductName
        self.Ver = Ver
        self.Date = Date
        self.Tile = Tile
        self.Obrit = Obrit
        self.ParaVer = ParaVer
        self.Target_path = Target_path
    def path(self):
        return '/standard/GCOM-C/GCOM-C.SGLI/L2.LAND.{}/{}/{}/{}/{}'.format(self.ProductName, self.Ver, self.Date[0:4], self.Date[4:6], self.Date[6:8])

    def filename(self):
        return 'GC1SG1_{}{}01D_T{}_L2SG_{}Q_{}00{}.h5'.format(self.Date, self.Obrit, self.Tile, self.ProductName, self.Ver, self.ParaVer)

    def download_file(self):
        remote_path = self.path()
        remote_file = self.filename()
        local_file = '{}{}'.format(self.Target_path, remote_file)
        return super().download(remote_path, local_file)
    def path(self):
        return '/standard/GCOM-C/GCOM-C.SGLI/L2.LAND.{}/{}/{}/{}/{}'.format(self.ProductName, self.Ver, self.Date[0:4], self.Date[4:6], self.Date[6:8])

    def filename(self):
        return 'GC1SG1_{}{}01D_T{}_L2SG_{}Q_{}00{}.h5'.format(self.Date, self.Obrit, self.Tile, self.ProductName, self.Ver, self.ParaVer)

    def download_file(self):
        remote_path = self.path()
        remote_file = self.filename()
        local_file = '{}{}'.format(self.Target_path, remote_file)
        download_status = super().download(remote_path, local_file)
        if download_status == 'y':
            return remote_file
        else:
            return "No Remote File"






def calc_sunpos(dtime,lat,lon):
    sun = ephem.Sun()
    obs = ephem.Observer()
    obs.date = dtime
    obs.lat = lat*math.pi/180.0
    obs.long = lon*math.pi/180.0
    sun.compute(obs)
    return np.degrees(sun.az),90.0-np.degrees(sun.alt)

def reporjection_GEO(filename,Band,Target_path):
    p = subprocess.Popen('/data01/people/liwei/Data/GCOM-C_RSRF/SGLI_geo_map_linux.exe {} -d Geometry_data/{} -o {} -r 0 -s 30'.format(filename,Band,Target_path[:-1]),shell=True)
    p.communicate()

def reporjection_IMAGE(filename,Band,Target_path):
    p = subprocess.Popen('/data01/people/liwei/Data/GCOM-C_RSRF/SGLI_geo_map_linux.exe {} -d Image_data/{} -o {} -s 30'.format(filename,Band,Target_path[:-1]),shell=True)
    p.communicate()  


def SGLI_2_AHI_GEO(filepath,lat,lon):  
    geotiff_da = rioxarray.open_rasterio(filepath)
    data = geotiff_da.interp(x=lon,y=lat,method="nearest")
    if data != -32768:
        data = data * 0.01
        return data
    else:
        return np.nan
    
def SGLI_2_AHI_Ref(filepath,lat,lon):  
    geotiff_da = rioxarray.open_rasterio(filepath)
    data = geotiff_da.interp(x=lon,y=lat,method="nearest")
    if data != 65535:
        data = data * 0.0001
        return data
    else:
        return np.nan    
    
def SGLI_2_AHI_TIME(filepath,lat,lon):
    geotiff_da = rioxarray.open_rasterio(filepath)
    data = geotiff_da.interp(x=lon,y=lat,method="nearest")
    if data != -32768.:
        data = data * 0.001
        return data
    else:
        return np.nan

def download_tile(date,save_tile_path,tile):
    g = SGLI_L2_Downloader(
        'galiwei ' ,
        'RSRF', 
        '3', 
         date , 
         tile, 
        'D' , 
        '1' ,
         save_tile_path
        )
    
def lonlat2tileidx(lat,lon):
    # vertical pixel count in one tile, horizontal pixel count in one tile
    lintile, coltile = 1200, 1200
    # vertical tile count, horizontal tile count
    vtilenum, htilenum = 18, 36
    # [deg/pixel]
    d = 180.0/lintile/vtilenum
    # from S-pole to N-pole
    NL = 180.0/d
    NP0 = 2*np.round(180.0/d)
    res1 = (90 - lat)/d-0.5
    V_idx = res1 // lintile
    # Y_idx = res1 - V_idx * lintile
    # GCOM-C use integer NPi
    NPi = round(NP0*np.cos(np.deg2rad(lat)))
    res2 = lon*NPi/360 - 0.5 + NP0/2
    H_idx = res2 // coltile
    # X_idx = res2 - H_idx * coltile
    return str(round(V_idx)).rjust(2,'0')  + str(round(H_idx)).rjust(2,'0') 

def SGLI_SA(d,lat,lon):
    SAA,SZA = calc_sunpos(d,lat,lon)
    return SAA,SZA

def Time_split_SGLI(time):
    YYYY = time.strftime('%Y')
    MM = time.strftime('%m')
    DD = time.strftime('%d')
    HH = time.strftime('%H')
    MIN = time.strftime('%M')
    date = YYYY + MM + DD
    return YYYY,MM,DD,HH,MIN,date

    
class SGLI_2_AHI:
    def __init__(self, path, date, tile, Band):
        self.path = path
        self.date = date
        self.tile = tile
        self.Band = Band

    def filepath(self):
        return '{}GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_{}.tif'.format(self.path, self.date, self.tile, self.Band)

    def get_data(self, lat, lon):
        filepath = self.filepath()
        geotiff_da = rioxarray.open_rasterio(filepath)
        data = geotiff_da.interp(x=lon,y=lat,method="nearest")
        return data
        
    def get_value(self, lat, lon):
        bands_Ref = ['Rp_PL01', 'Rp_PL02', 'Rs_PI01', 'Rs_PI02', 'Rs_SW01', 'Rs_SW02', 'Rs_SW03', 'Rs_SW04', 'Rs_VN01', 'Rs_VN02', 'Rs_VN03', 'Rs_VN04', 'Rs_VN05', 'Rs_VN06', 'Rs_VN07', 'Rs_VN08', 'Rs_VN08P', 'Rs_VN09', 'Rs_VN10', 'Rs_VN11', 'Rs_VN11P', 'SWR', 'Tau_500', 'Tb_TI01', 'Tb_TI02']
        bands_GEO = ['Sensor_azimuth', 'Sensor_azimuth_IR', 'Sensor_azimuth_PL', 'Sensor_zenith', 'Sensor_zenith_IR', 'Sensor_zenith_PL', 'Solar_azimuth', 'Solar_azimuth_PL', 'Solar_zenith', 'Solar_zenith_PL']
        bands_TIME = ['Obs_time', 'Obs_time_PL']
        data = self.get_data(lat,lon)
        if self.Band in bands_Ref:
            if data != 65535:
                data = data * 0.0001
                return data
            else:
                return np.nan
        elif self.Band in bands_GEO:
            if data != -32768:
                data = data * 0.01
                return data
            else:
                return np.nan
        elif self.Band in bands_TIME:
            if data != -32768.:
                data = data * 0.001
                return data
            else:
                return np.nan   

reporjection_file_path = '/data01/people/liwei/Data/GCOM-C_RSRF/reporjection/'
tile_file_path = '/data01/people/liwei/Data/GCOM-C_RSRF/tile/'

def read_SGLI(date,AHI_lat,AHI_lon,lat_idx,lon_idx):
 # 获取该经纬度的Tile号
                # print(lat_idx,lon_idx) 

    tile = lonlat2tileidx(AHI_lat[lat_idx],AHI_lon[lon_idx])
    
    # print(tile) 
#                 # SGLI 文件名
    SGLI_tile_filename = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001.h5'.format(date,tile)
    SGLI_VZA_filename = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_Sensor_zenith.tif'.format(date,tile)
    SGLI_VAA_filename = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_Sensor_azimuth.tif'.format(date,tile)
    SGLI_TIME_filename = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_Obs_time.tif'.format(date,tile)
    # SGLI_VZA_filename_PL = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_Sensor_zenith_PL.tif'.format(date,tile)
    # SGLI_VAA_filename_PL = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_Sensor_azimuth_PL.tif'.format(date,tile)
    # SGLI_TIME_filename_PL = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_Obs_time_PL.tif'.format(date,tile)


    SGLI_REF_VN04_filename = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_Rs_VN04.tif'.format(date,tile)
    SGLI_REF_VN05_filename = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_Rs_VN05.tif'.format(date,tile)
    SGLI_REF_VN07_filename = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_Rs_VN07.tif'.format(date,tile)
    SGLI_REF_VN10_filename = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_Rs_VN10.tif'.format(date,tile)
    SGLI_REF_SW03_filename = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_Rs_SW03.tif'.format(date,tile)
    SGLI_REF_SW04_filename = 'GC1SG1_{}D01D_T{}_L2SG_RSRFQ_3001_Rs_SW04.tif'.format(date,tile)

    if os.path.exists(reporjection_file_path + SGLI_VAA_filename) and os.path.exists(reporjection_file_path + SGLI_VZA_filename) and os.path.exists(reporjection_file_path + SGLI_TIME_filename):
        # print('1')
        # 读取该像素经纬度SGLI的VZA和VAA
        SGLI_VZA_PL = SGLI_2_AHI_GEO(reporjection_file_path + SGLI_VZA_filename,AHI_lat[lat_idx],AHI_lon[lon_idx])
        SGLI_VAA_PL = SGLI_2_AHI_GEO(reporjection_file_path + SGLI_VAA_filename,AHI_lat[lat_idx],AHI_lon[lon_idx])

        if not np.isnan(SGLI_VZA_PL) and not np.isnan(SGLI_VAA_PL):
            if SGLI_VAA_PL < 0:
                SGLI_VAA_PL = SGLI_VAA_PL + 360
        # 计算该像素经纬度SGLI的SZA和SAA

            SGLI_Obs_TIME = SGLI_2_AHI_TIME(reporjection_file_path + SGLI_TIME_filename,AHI_lat[lat_idx],AHI_lon[lon_idx])
            # print(SGLI_Obs_TIME.values[0])
            # print(SGLI_Obs_TIME)
            if not np.isnan(SGLI_Obs_TIME):
                # print(SGLI_Obs_TIME)
                SGLI_MIN = int(round(math.modf(SGLI_Obs_TIME)[0],3)*60)
                SGLI_HH = int(math.modf(SGLI_Obs_TIME)[1])
                if SGLI_HH <= 23 and SGLI_MIN >= 0 and SGLI_MIN <= 59:
                    dtime = dt.datetime.strptime(date+str(SGLI_HH)+str(SGLI_MIN), "%Y%m%d%H%M")
                    SGLI_SAA_PL,SGLI_SZA_PL = SGLI_SA(dtime,AHI_lat[lat_idx],AHI_lon[lon_idx])
                    SGLI_Ref_VN04 = SGLI_2_AHI_Ref(reporjection_file_path + SGLI_REF_VN04_filename,AHI_lat[lat_idx],AHI_lon[lon_idx])
                    SGLI_Ref_VN05 = SGLI_2_AHI_Ref(reporjection_file_path + SGLI_REF_VN05_filename,AHI_lat[lat_idx],AHI_lon[lon_idx])
                    SGLI_Ref_VN07 = SGLI_2_AHI_Ref(reporjection_file_path + SGLI_REF_VN07_filename,AHI_lat[lat_idx],AHI_lon[lon_idx])
                    SGLI_Ref_VN10 = SGLI_2_AHI_Ref(reporjection_file_path + SGLI_REF_VN10_filename,AHI_lat[lat_idx],AHI_lon[lon_idx])
                    SGLI_Ref_SW03 = SGLI_2_AHI_Ref(reporjection_file_path + SGLI_REF_SW03_filename,AHI_lat[lat_idx],AHI_lon[lon_idx])
                    SGLI_Ref_SW04 = SGLI_2_AHI_Ref(reporjection_file_path + SGLI_REF_SW04_filename,AHI_lat[lat_idx],AHI_lon[lon_idx])
                    # 读取AHI的SZA、SAA
                    SGLI_RAA_PL = abs(SGLI_SAA_PL - SGLI_VAA_PL.data[0])
                    SGLI_RAA_PL = np.where(SGLI_RAA_PL>180,360-SGLI_RAA_PL,SGLI_RAA_PL)
                    
                    if not np.isnan(SGLI_Ref_VN04):
                        SGLI_Ref_VN04 = SGLI_Ref_VN04.values[0]
                    if not np.isnan(SGLI_Ref_VN05):
                        SGLI_Ref_VN05 = SGLI_Ref_VN05.values[0]
                    if not np.isnan(SGLI_Ref_VN07):
                        SGLI_Ref_VN07 = SGLI_Ref_VN07.values[0]
                    if not np.isnan(SGLI_Ref_VN10):
                        SGLI_Ref_VN10 = SGLI_Ref_VN10.values[0]
                    if not np.isnan(SGLI_Ref_SW03):
                        SGLI_Ref_SW03 = SGLI_Ref_SW03.values[0]
                    if not np.isnan(SGLI_Ref_SW04):
                        SGLI_Ref_SW04 = SGLI_Ref_SW04.values[0]
                        
                    return SGLI_VZA_PL.values[0],SGLI_SZA_PL,SGLI_RAA_PL,SGLI_SAA_PL,SGLI_VAA_PL.values[0],\
                            SGLI_Ref_VN04,SGLI_Ref_VN05,SGLI_Ref_VN07,SGLI_Ref_VN10,SGLI_Ref_SW03,SGLI_Ref_SW04

                else:
                    return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
            else:
                return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        else:
            return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    else:
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
