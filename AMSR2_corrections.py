# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 17:04:47 2022

@author: Pau
"""

import ftplib
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


def TB_correction(TB_meas, TB_model, RTM_base):
    Delta_TB = RTM_base - TB_model 
    TB_ = TB_meas + Delta_TB
    return TB_


def ratio(sigma_meas, sigma_model):
    R = (sigma_meas - sigma_model) / (sigma_meas + sigma_model)
    return R


def identify_size(inc):
    diff = inc[1:] - inc[:-1]
    size_Y = np.where( diff < 0 )[0][0] + 1
    size_X = int(np.shape(inc)[0] / size_Y)
    return size_X, size_Y



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


# RADIOMETRIC data from AMSR2
bt_6_9_h = ds['btemp_6_9h'].values
bt_10_7_h = ds['btemp_10_7h'].values
bt_18_7_h = ds['btemp_18_7h'].values
bt_23_8_h = ds['btemp_23_8h'].values
bt_36_5_h = ds['btemp_36_5h'].values

D_mead_TB = np.dstack((bt_6_9_h, bt_10_7_h, bt_18_7_h, bt_23_8_h, bt_36_5_h))
orig_size = np.shape(bt_6_9_h)

bt_18_7_v = ds['btemp_18_7v'].values
bt_36_5_v = ds['btemp_36_5v'].values


# RADIOMETRIC CORRECTION: MODEL FOR AMSR2 USING ERAS05 DATASET
u = ds['u10m_rotated'].values   # wind azimuth
v = ds['v10m_rotated'].values   # wind range
W_pow = np.sqrt(u**2 + v**2)
V = ds['tcwv'].values 
L = ds['tclw'].values
Ta = ds['t2m'].values

Ts=275.0
Ti=260.0
Ti_amsrv = Ti*np.ones(8)
Ti_amsrh = Ti*np.ones(8)
c_ice=0.0
e_icev_base=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
e_iceh_base=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])

e_icev=np.ones(8)*0.9
e_iceh=np.ones(8)*0.8


# SIC
sic = ica.nasa(bt_18_7_v, bt_18_7_h, bt_36_5_v)



idx_interest = np.array([1,3,5,7,9])
AMSR2_model = np.empty((W_pow.shape[0], W_pow.shape[1], 12))
AMSR2_model_sic = np.empty((W_pow.shape[0], W_pow.shape[1], 12))
RTM_sic = np.empty((W_pow.shape[0], W_pow.shape[1], 12))

RTM_base = amsr(0,0,0,np.mean(Ta),Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev_base,e_iceh_base,55)

for i in range(W_pow.shape[0]):
    for j in range(W_pow.shape[1]):
        AMSR2_model[i,j,:] = amsr(V[i,j],W_pow[i,j],L[i,j],Ta[i,j],Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev_base,e_iceh_base,55)
        AMSR2_model_sic[i,j,:] = amsr(V[i,j],W_pow[i,j],L[i,j],Ta[i,j],Ts,Ti_amsrv,Ti_amsrh,sic[i,j],e_icev,e_iceh,55)
        RTM_sic[i,j,:] = amsr(0,0,0,np.mean(Ta),Ts,Ti_amsrv,Ti_amsrh,sic[i,j],e_icev,e_iceh,55)

TB_corr_base = TB_correction(D_mead_TB, AMSR2_model[:,:,idx_interest], RTM_base[idx_interest])
TB_corr_sic = TB_correction(D_mead_TB, AMSR2_model[:,:,idx_interest], RTM_sic[:,:,idx_interest])

## PLOTTING ##

f = plt.figure()
plt.imshow(bt_36_5_h, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
plt.suptitle('AMSR2 measurements TBH 36.5GHz, S{}'.format(scene))
plt.savefig('AMSR_meas_s{}.png'.format(scene), format='PNG', dpi=300)

f = plt.figure()
plt.imshow(AMSR2_model[:,:,9], cmap='RdBu_r')
quak = plt.colorbar(extend='both')
plt.suptitle('AMSR2 model TBH 36.5GHz SIC = 0, S{}'.format(scene))
plt.savefig('AMSR_model_s{}.png'.format(scene), format='PNG', dpi=300)

f = plt.figure()
plt.imshow(AMSR2_model_sic[:,:,9], cmap='RdBu_r')
quak = plt.colorbar(extend='both')
plt.suptitle('AMSR2 model TBH 36.5GHz SIC != 0, S{}'.format(scene))
plt.savefig('AMSR_modelsic_s{}.png'.format(scene), format='PNG', dpi=300)

f = plt.figure()
plt.imshow(RTM_sic[:,:,5], cmap='RdBu_r')
quak = plt.colorbar(extend='both')
plt.suptitle('AMSR2 baseline 36.5GHz SIC != 0, S{}'.format(scene))
plt.savefig('AMSR_baselinesic_s{}.png'.format(scene), format='PNG', dpi=300)

f = plt.figure()
plt.imshow(TB_corr_base[:,:,4], cmap='RdBu_r')
quak = plt.colorbar(extend='both')
plt.suptitle('AMSR2 corrected TBH 36.5GHz SIC = 0, S{}'.format(scene))
plt.savefig('AMSR_corr_s{}.png'.format(scene), format='PNG', dpi=300)

f = plt.figure()
plt.imshow(TB_corr_sic[:,:,4], cmap='RdBu_r')
quak = plt.colorbar(extend='both')
plt.suptitle('AMSR2 corrected TBH 36.5GHz SIC != 0, S{}'.format(scene))
plt.savefig('AMSR_corrsic_s{}.png'.format(scene), format='PNG', dpi=300)




