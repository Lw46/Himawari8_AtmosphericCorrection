from joblib import Parallel, delayed
import numpy as np
from sklearn.linear_model import LinearRegression

def RTK (sza,vza,raa):
    cos_xi = np.cos(np.radians(sza)) * np.cos(np.radians(vza)) + np.sin(np.radians(sza)) * np.sin(np.radians(vza))* np.cos(np.radians(raa))
    xi = np.arccos(cos_xi)
    rtk = (((np.pi/2 - xi) * cos_xi + np.sin(xi)) / (np.cos(np.radians(sza)) + np.cos(np.radians(vza)))) - np.pi/4
    return rtk

def RJN(sza,vza,raa):
    cos_sza = np.cos(np.radians(sza))
    cos_vza = np.cos(np.radians(vza))
    cos_raa = np.cos(np.radians(raa))

    sin_sza = np.sin(np.radians(sza))
    sin_vza = np.sin(np.radians(vza))
    sin_raa = np.sin(np.radians(raa))

    tan_sza = np.tan(np.radians(sza))
    tan_vza = np.tan(np.radians(vza))
    _sza = np.arctan(tan_sza)
    _vza = np.arctan(tan_vza)

    cos__sza = np.cos(_sza)
    cos__vza = np.cos(_vza)
    sin__sza = np.sin(_sza)
    sin__vza = np.sin(_vza)
    cos__xi = cos__sza * cos__vza + sin__sza * sin__vza * cos_raa
    D = np.sqrt(np.square(tan_sza) + np.square(tan_vza) - 2 * tan_sza * tan_vza * cos_raa)
    rjn = 1/2*np.pi * ((np.pi - np.radians(raa)) * cos_raa + sin_raa) * tan_sza * tan_vza - 1/np.pi * (tan_sza + tan_vza + D)
    return rjn

def LSR_MaxMin(sza,vza,raa):
    cos_sza = np.cos(np.radians(sza))
    cos_vza = np.cos(np.radians(vza))
    cos_raa = np.cos(np.radians(raa))

    sin_sza = np.sin(np.radians(sza))
    sin_vza = np.sin(np.radians(vza))
    sin_raa = np.sin(np.radians(raa))

    tan_sza = np.tan(np.radians(sza))
    tan_vza = np.tan(np.radians(vza))

    sec_sza = 1 / cos_sza
    sec_vza = 1 / cos_vza

    _sza = np.arctan(tan_sza)
    _vza = np.arctan(tan_vza)

    cos__sza = np.cos(_sza)
    cos__vza = np.cos(_vza)
    sin__sza = np.sin(_sza)
    sin__vza = np.sin(_vza)
    sec__sza = 1 / cos__sza
    sec__vza = 1 / cos__vza

    cos__xi = cos__sza * cos__vza + sin__sza * sin__vza * cos_raa
    D = np.sqrt(np.square(tan_sza) + np.square(tan_vza) - 2 * tan_sza * tan_vza * cos_raa)
    cos_t = (2 * (np.sqrt(np.square(D) + np.square(tan_sza * tan_vza * sin_raa)))) / (sec__sza + sec__vza)
    
    cos_t = np.where(cos_t>1,1,cos_t)
    cos_t = np.where(cos_t<-1,-1,cos_t)
    
    
    t = np.arccos(cos_t)
    O = (1/np.pi) * (t - np.sin(t) * cos_t) *  (sec__sza + sec__vza)
    lsr = O - sec__sza - sec__vza + 0.5 * (1 + cos__xi) * sec__sza * sec__vza
    return lsr

def LSR(sza,vza,raa):
    cos_sza = np.cos(np.radians(sza))
    cos_vza = np.cos(np.radians(vza))
    cos_raa = np.cos(np.radians(raa))

    sin_sza = np.sin(np.radians(sza))
    sin_vza = np.sin(np.radians(vza))
    sin_raa = np.sin(np.radians(raa))

    tan_sza = np.tan(np.radians(sza))
    tan_vza = np.tan(np.radians(vza))

    sec_sza = 1 / cos_sza
    sec_vza = 1 / cos_vza

    _sza = np.arctan(tan_sza)
    _vza = np.arctan(tan_vza)

    cos__sza = np.cos(_sza)
    cos__vza = np.cos(_vza)
    sin__sza = np.sin(_sza)
    sin__vza = np.sin(_vza)
    sec__sza = 1 / cos__sza
    sec__vza = 1 / cos__vza

    cos__xi = cos__sza * cos__vza + sin__sza * sin__vza * cos_raa
    D = np.sqrt(np.square(tan_sza) + np.square(tan_vza) - 2 * tan_sza * tan_vza * cos_raa)
    cos_t = (2 * (np.sqrt(np.square(D) + np.square(tan_sza * tan_vza * sin_raa)))) / (sec__sza + sec__vza)
    
    t = np.arccos(cos_t)
    O = (1/np.pi) * (t - np.sin(t) * cos_t) *  (sec__sza + sec__vza)
    lsr = O - sec__sza - sec__vza + 0.5 * (1 + cos__xi) * sec__sza * sec__vza
    return lsr

def LSD(sza,vza,raa):
    cos_sza = np.cos(np.radians(sza))
    cos_vza = np.cos(np.radians(vza))
    cos_raa = np.cos(np.radians(raa))

    sin_sza = np.sin(np.radians(sza))
    sin_vza = np.sin(np.radians(vza))
    sin_raa = np.sin(np.radians(raa))

    tan_sza = np.tan(np.radians(sza))
    tan_vza = np.tan(np.radians(vza))

    sec_sza = 1 / cos_sza
    sec_vza = 1 / cos_vza

    _sza = np.arctan(tan_sza)
    _vza = np.arctan(tan_vza)

    cos__sza = np.cos(_sza)
    cos__vza = np.cos(_vza)
    sin__sza = np.sin(_sza)
    sin__vza = np.sin(_vza)
    sec__sza = 1 / cos__sza
    sec__vza = 1 / cos__vza

    cos__xi = cos__sza * cos__vza + sin__sza * sin__vza * cos_raa
    D = np.sqrt(np.square(tan_sza) + np.square(tan_vza) - 2 * tan_sza * tan_vza * cos_raa)
    cos_t = (2 * (np.sqrt(np.square(D) + np.square(tan_sza * tan_vza * sin_raa)))) / (sec__sza + sec__vza)
    t = np.arccos(cos_t)
    O = (1/np.pi) * (t - np.sin(t) * cos_t) *  (sec__sza + sec__vza)
    lsd = ((1 + cos__xi) * sec__vza) / (sec__sza + sec__vza - O) -2
    return lsd

def get_new_kgeo(kernel,i,j):
    
    a = np.array(kernel)[:,i,j]
    s_idx ,e_idx = a.size // 2,a.size // 2
    
    while s_idx != -1:
        if not np.isnan(a[s_idx]):
            s_idx -= 1
        else:
            break
    while e_idx != a.size:
        if not np.isnan(a[e_idx]):
            e_idx += 1
        else:
            break  
            
    max_idx = np.nanargmax(a)
   
    
    if s_idx == -1 and e_idx != a.size:
        x = np.arange(max_idx + 2 ,e_idx ,1)
        y = a[max_idx + 2:e_idx]
        x2 = np.arange(e_idx,a.size,1)
        p2 = np.polyfit(x, y, 2)
        yvals2 = np.polyval(p2,x2).tolist()
        
        return a[0:e_idx].tolist() + yvals2
        
    elif s_idx != -1 and e_idx == a.size:
        x = np.arange(s_idx+1 ,max_idx -2 ,1)
        y =a[s_idx+1:max_idx -2]
        x1 = np.arange(0,s_idx+1,1)
        p1 = np.polyfit(x, y, 2)
        yvals1 = np.polyval(p1,x1).tolist()
        return yvals1 + a[s_idx+1:].tolist()
        
    elif s_idx != -1 and e_idx != a.size:
        x = np.arange(s_idx+1 ,max_idx -2 ,1)
        y =a[s_idx+1:max_idx -2]
        x1 = np.arange(0,s_idx+1,1)
        p1 = np.polyfit(x, y, 2)
        yvals1 = np.polyval(p1,x1).tolist()

        x = np.arange(max_idx + 2 ,e_idx ,1)
        y = a[max_idx + 2:e_idx]
        x2 = np.arange(e_idx,a.size,1)
        p2 = np.polyfit(x, y, 2)
        yvals2 = np.polyval(p2,x2).tolist()
        return yvals1 + a[s_idx+1:e_idx].tolist() + yvals2


def LSR_fit(lsr):
    Kresult = Parallel(n_jobs=15)(delayed(get_new_kgeo)(lsr,i,j) for i in range(12) for j in range(12))
    result = np.array(Kresult).T.reshape(49,12,12)
    return result

import time

def fitting(ref,dg,dv,i,j):
    r = ref[:,i,j].reshape(-1,1)
    cm = ~np.isnan(r)
    r = r[cm].reshape(-1,1)
    kv = dv[:,i,j].reshape(-1,1)[cm].reshape(-1,1)
    kg = dg[:,i,j].reshape(-1,1)[cm].reshape(-1,1)

    # 检查是否存在空值，并去除
    if not (np.isnan(kg) | np.isnan(kv)).any():
        k = np.hstack((kv, kg))
    else:
        return np.nan, np.nan, np.nan

    # 检查数据量是否大于40
    if r.size > 60:
        linreg = LinearRegression()
        model = linreg.fit(k, r)
        fi = linreg.intercept_[0]
        fv, fg = linreg.coef_[0]
        return fi, fv, fg
    else:
        return np.nan, np.nan, np.nan

SBAF_Offset_OS = [0.008,0.006,0.002,0.0002,0.0015,0]
SBAF_Slope_OS = [0.95,0.927,0.896,0.984,0.987,1]

SBAF_Offset_Grassland = [0.0055,0.0007,0.0027,0,-0.0017,0]
SBAF_Slope_Grassland = [0.973,0.993,0.909,0.984,0.995,1]

SBAF_Offset_Cropland = [0.007,0.0005,0.0055,0.0005,0.0005,0]
SBAF_Slope_Cropland = [0.979,1,0.906,0.976,1.001,1]

SBAF_Offset_EBF = [0.0054,-0.003,0.004,0.0006,0.002,0]
SBAF_Slope_EBF = [1.042,1.071,1.007,0.964,1.019,1]

SBAF_Offset_Savanna = [0.0069,0.004,0.0053,0.0001,0.0035,0]
SBAF_Slope_Savanna = [0.973,0.962,0.880,0.977,0.985,1]

def get_sbaf(LC,band):
    if LC == 'EBF':
        return SBAF_Slope_EBF[band-1],SBAF_Offset_EBF[band-1]
    elif LC == 'Open_Shrubland':
        return SBAF_Slope_OS[band-1],SBAF_Offset_OS[band-1]
    elif LC == 'Grassland':
        return SBAF_Slope_Grassland[band-1],SBAF_Offset_Grassland[band-1]
    elif LC == 'Cropland':
        return SBAF_Slope_Cropland[band-1],SBAF_Offset_Cropland[band-1]
    elif LC == 'Savanna':
        return SBAF_Slope_Savanna[band-1],SBAF_Offset_Savanna[band-1]