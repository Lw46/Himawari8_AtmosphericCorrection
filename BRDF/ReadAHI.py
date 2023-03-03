import numpy as np
import os
import cv2
import threading
import concurrent.futures

SZA_PATH = '/data01/GEO/INPUT/ANGLE/Solar_Zenith_Angle_u2/'
SAA_PATH = '/data01/GEO/INPUT/ANGLE/Solar_Azimuth_Angle_u2/'
SR_PATH = '/data01/people/liwei/AC_Result/'


def read_SolarAngle(filename):

    SA = np.memmap(filename, dtype='u2', mode='r', shape=(3000, 3000)) / 100
    SA=cv2.resize(np.array(SA,dtype='float64'),(12000,12000),interpolation=cv2.INTER_NEAREST)
    return SA

def read_solar_angle(date):
    AHI_date = date[4:11]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        file_list = [SZA_PATH + 'AHI_SZA_2018{}5.dat'.format(AHI_date), SAA_PATH + 'AHI_SAA_2018{}5.dat'.format(AHI_date)]
        results = [executor.submit(read_SolarAngle, file) for file in file_list]
    Result = [result.result() for result in results]
    return Result



def read_SR(filename):
    if os.path.exists(filename):
        if filename[-5] == '5' or filename[-5] == '6':
            SR = np.memmap(filename, dtype='int16', mode='r', shape=(6000, 6000))
            SR=cv2.resize(np.array(SR,dtype='float64'),(12000,12000),interpolation=cv2.INTER_NEAREST)
        else:
            SR = np.memmap(filename, dtype='int16', mode='r', shape=(12000, 12000))
        return SR
    else:
        return np.full((12000,12000),np.nan)

def read_sr(date):
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        file_list = [SR_PATH + '{}_AC/{}_b01.dat'.format(date,date), \
                     SR_PATH + '{}_AC/{}_b02.dat'.format(date,date), \
                     SR_PATH + '{}_AC/{}_b03.dat'.format(date,date), \
                     SR_PATH + '{}_AC/{}_b04.dat'.format(date,date), \
                     SR_PATH + '{}_AC/{}_b05.dat'.format(date,date), \
                     SR_PATH + '{}_AC/{}_b06.dat'.format(date,date)]
        results = [executor.submit(read_SR, file) for file in file_list]
    Result = [result.result() for result in results]
    return Result




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



def Time_split_AHI(time):
    date = time.strftime("%Y%m%d%H%M")
    YYYY, MM, DD, HH, MIN = date[0:4], date[4:6], date[6:8], date[8:10], date[10:12]
    return YYYY,MM,DD,HH,MIN,date

LAND_MASK = read_Landmask(0.01)

def read_cloudmask(date):
    file_path = '/data01/people/liwei/AHIcm_algo_v0/cloudmask/{}/AHIcm.v0.{}.dat'.format(date[:6],date)
    if os.path.exists(file_path):
        cloudmask = np.memmap(file_path, dtype='<f4', mode='r', shape=(6000, 6000))
        cloudmask = cv2.resize(cloudmask,(12000,12000),interpolation=cv2.INTER_NEAREST)
        c_o_mask = LAND_MASK + cloudmask
        c_o_mask_1 = np.where(c_o_mask==2,1,0).astype(bool)
    else :
        c_o_mask_1 = np.full((12000,12000),0).astype(bool)
    return c_o_mask_1