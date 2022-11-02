# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 12:02:39 2022

@author: Pau
"""

import xarray as xr
import os,glob,sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from amsrrtm2 import amsr
from cmod5n import cmod5n_forward
import ic_algs as ica



def normalize(X):
    X_max = np.nanmax(X)
    X_min = np.nanmin(X)
    X_out = np.divide((X - X_min), (X_max - X_min))
    return np.nan_to_num(X_out)


def invNorm(X):
    X_max = np.nanmax(X)
    X_min = np.nanmin(X)
    X_out = np.divide((X_max - X), (X_max - X_min))
    return np.nan_to_num(X_out)


def TB_correction(TB_meas, TB_model, RTM_base):
    Delta_TB = RTM_base - TB_model 
    TB_ = TB_meas + Delta_TB
    return TB_


def scatterCorrection(sigma_meas, sigma_model, flag_norm = 1):
    if flag_norm:
        sigma_corr = sigma_meas - sigma_model
    else:
        sigma_corr = normalize(sigma_meas) - normalize(sigma_model)
    return sigma_corr


def ratio(sigma_meas, sigma_model):
    R = (sigma_meas - sigma_model) / (sigma_meas + sigma_model)
    return R


def identify_size(inc):
    diff = inc[1:] - inc[:-1]
    size_Y = np.where( diff < 0 )[0][0] + 1
    size_X = int(np.shape(inc)[0] / size_Y)
    return size_X, size_Y


def nanAverageFitler(img):
    kernel = np.array([[1, 1, 1],   #3x3 kernel
                       [1, 1, 1],
                       [1, 1, 1]])
    (dim_x, dim_y) = np.shape(img)
    
    imgFiltered = np.zeros(img.shape) #the new image. the same size as the image I will filter
   
    for i in range(1,dim_x-1): #the range starts from 1 to avoid the column and row of zeros, and ends before the last col and row of zeros
         for j in range(1,dim_y-1):
             imagen_entry = img[i-1:i+2, j-1:j+2]
             average = np.nanmean(imagen_entry*kernel)    #Matrix 3x3 is filled with the elements around each [i, j] entry of the array
             imgFiltered[i, j] = average
    return imgFiltered
    
    

## Get scene
dataset_path = "../Datasets/"
images_path = "../Image_outputs/MASKS/"

scene = 3 

if scene == 2:
    ds = xr.open_dataset(dataset_path + 'S1A_EW_GRDM_1SDH_20180730T180326_20180730T180426_023027_027FE2_D0F8_icechart_dmi_201807301805_CentralEast_RIC.nc')   
elif scene == 7:
    ds = xr.open_dataset(dataset_path + 'S1A_EW_GRDM_1SDH_20190507T101118_20190507T101218_027120_030E8D_4E4F_icechart_cis_SGRDINFLD_20190507T1011Z_pl_a.nc')   
elif scene == 3:
    ds = xr.open_dataset(dataset_path + 'S1A_EW_GRDM_1SDH_20190216T205218_20190216T205323_025960_02E45A_A52C_icechart_dmi_201902162055_CapeFarewell_RIC.nc')   

## AMSR2 mask
HH = ds['nersc_sar_primary'].values


#orig_size = np.shape(bt_6_9_h)


# RADIOMETRIC CORRECTION: MODEL FOR AMSR2 USING ERAS05 DATASET
u = ds['u10m_rotated'].values   # wind azimuth
v = ds['v10m_rotated'].values   # wind range
W_pow = np.sqrt(u**2 + v**2)


## CMOD5N MASK ##
incidence = ds['sar_grid_incidenceangle'].values
phi = np.arccos(u/W_pow)*180/np.pi
size_inc = identify_size(incidence)
inc_2D = incidence.reshape(size_inc[0], size_inc[1])

## Reshape cmod5n model parameters to SAR image size grid (interpolate)
x = np.linspace(0,np.shape(inc_2D)[1]-1,np.shape(inc_2D)[1])
y = np.linspace(0,np.shape(inc_2D)[0]-1,np.shape(inc_2D)[0])
f_inc = interpolate.interp2d(x, y, inc_2D, kind='linear')

x = np.linspace(0,np.shape(inc_2D)[1]-1,np.shape(W_pow)[1])
y = np.linspace(0,np.shape(inc_2D)[0]-1,np.shape(W_pow)[0])
inc_grid_2k = f_inc(x,y)

cmod5n_values = cmod5n_forward(W_pow, phi, inc_grid_2k)

x = np.linspace(0,np.shape(cmod5n_values)[1]-1,np.shape(cmod5n_values)[1])
y = np.linspace(0,np.shape(cmod5n_values)[0]-1,np.shape(cmod5n_values)[0])
f_cmod = interpolate.interp2d(x, y, cmod5n_values, kind='linear')

x = np.linspace(0,np.shape(cmod5n_values)[1]-1,np.shape(HH)[1])
y = np.linspace(0,np.shape(cmod5n_values)[0]-1,np.shape(HH)[0])
cmod5n_reshape = f_cmod(x,y)


### CORRECT SCATTER MEASUREMENTS ###
# HH measurements in logarithmic scale
# CMOD5N values linear normalized scale
### SIC contricution ###
bt_18_7_h = ds['btemp_18_7h'].values
bt_18_7_v = ds['btemp_18_7v'].values
bt_36_5_v = ds['btemp_36_5v'].values

sic = ica.nasa(bt_18_7_v, bt_18_7_h, bt_36_5_v)

x = np.linspace(0,np.shape(sic)[1]-1,np.shape(sic)[1])
y = np.linspace(0,np.shape(sic)[0]-1,np.shape(sic)[0])
f_sic = interpolate.interp2d(x, y, sic, kind='linear')

x = np.linspace(0,np.shape(sic)[1]-1,np.shape(HH)[1])
y = np.linspace(0,np.shape(sic)[0]-1,np.shape(HH)[0])
sic_reshape = f_sic(x,y)

idx_mask = np.where(sic_reshape < 0.10)
mask_sic = sic_reshape*0
mask_sic[idx_mask[0], idx_mask[1]] = 1


# Try to transform HH to Linear normalized scale, correct it, and back to log scale --> Scale 10
HHx = HH
idx = np.argwhere(np.isnan(HH))
HHx[idx[:,0], idx[:,1]] = np.nanmean(HH)

HHx_lin = 10**(HHx/10)
HHx_corr = HHx_lin - np.multiply(cmod5n_reshape, mask_sic)
HHx_corrv2 = HHx_corr - np.nanmin(HHx_corr) + 0.001
HHx_corr_logv2 = 10*np.log10(HHx_corrv2)#HH_corr_log_norm = normalize(HH_corr_log)
#HHx_corr_log = 10*np.log10(HHx_corr)
#ratio_m = ratio(HHx_lin, cmod5n_reshape)
#ratio_log = 10*np.log10(ratio_m)

'''
plt.imshow(HHx_lin, cmap='RdBu_r')
quak = plt.colorbar(extend='both')

plt.imshow(HHx_corr, cmap='RdBu_r')
quak = plt.colorbar(extend='both')

plt.imshow(HHx_corrv2, cmap='RdBu_r')
quak = plt.colorbar(extend='both')

plt.imshow(HHx_corr_log, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
'''

q995 = np.quantile(HHx_corr_logv2,0.995)
q005 = np.quantile(HHx_corr_logv2,0.005)

HHx_corr_logv2[HHx_corr_logv2 > q995] = q995
HHx_corr_logv2[HHx_corr_logv2 < q005] = q005

f = plt.figure()
plt.imshow(HHx_corr_logv2, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
plt.suptitle('HH (10*log10())- S{}'.format(scene))
plt.savefig('scatter_10log10_s{}.png'.format(scene), format='PNG', dpi=300)


# Try to transform HH to Linear normalized scale, correct it, and back to log scale --> Scale 20
HHx = HH
idx = np.argwhere(np.isnan(HH))
HHx[idx[:,0], idx[:,1]] = np.nanmean(HH)

HHx_lin = 10**(HHx/20)
HHx_corr = HHx_lin - np.multiply(cmod5n_reshape, mask_sic)
HHx_corrv2 = HHx_corr - np.nanmin(HHx_corr) + 0.001
HHx_corr_logv2 = 20*np.log10(HHx_corrv2)#HH_corr_log_norm = normalize(HH_corr_log)
#HHx_corr_log = 10*np.log10(HHx_corr)
#ratio_m = ratio(HHx_lin, cmod5n_reshape)
#ratio_log = 10*np.log10(ratio_m)

f = plt.figure()
plt.imshow(HH, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
plt.suptitle('HH - S{}'.format(scene))
plt.savefig('HH_s{}.png'.format(scene), format='PNG', dpi=300)

q995 = np.quantile(HHx_corr_logv2,0.995)
q005 = np.quantile(HHx_corr_logv2,0.005)

HHx_corr_logv2[HHx_corr_logv2 > q995] = q995
HHx_corr_logv2[HHx_corr_logv2 < q005] = q005

f = plt.figure()
plt.imshow(HHx_corr_logv2, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
plt.suptitle('HH (20*log10())- S{}'.format(scene))
plt.savefig('scatter_20log10_s{}.png'.format(scene), format='PNG', dpi=300)


'''
f = plt.figure()
plt.imshow(HH - 20*np.log10(cmod5n_reshape), cmap='RdBu_r')
quak = plt.colorbar(extend='both')
plt.suptitle('AMSR2 measurements TBH 36.5GHz, S{}'.format(scene))
plt.savefig('AMSR_meas_s{}.png'.format(scene), format='PNG', dpi=300)
'''

f = plt.figure()
plt.imshow(mask_sic, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
plt.suptitle('Mask_sic - S{}'.format(scene))
plt.savefig('mask_sic_s{}.png'.format(scene), format='PNG', dpi=300)
