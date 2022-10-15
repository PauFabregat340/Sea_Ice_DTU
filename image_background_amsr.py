# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 20:01:02 2022

@author: Pau
"""


import ftplib
import xarray as xr
import os,glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from cmod5n import cmod5n_forward
from amsrrtm2 import amsr


def normalize(X):
    X_max = np.nanmax(X)
    X_min = np.nanmin(X)
    X_out = np.divide((X - X_min), (X_max - X_min))
    return np.nan_to_num(X_out)


def TB_correction(TB_meas, TB_model, RTM_base):
    Delta_TB = RTM_base - TB_model 
    TB_ = TB_meas + Delta_TB
    return TB_


scene = 4
## Get scene
# Scene 04
if scene ==4:
    ds = xr.open_dataset('S1A_EW_GRDM_1SDH_20190507T101118_20190507T101218_027120_030E8D_4E4F_icechart_cis_SGRDINFLD_20190507T1011Z_pl_a.nc')   
elif scene == 14:
    ds = xr.open_dataset('S1B_EW_GRDM_1SDH_20180424T212356_20180424T212505_010631_013669_E657_icechart_cis_SGRDINFLD_20180424T2121Z_pl_a.nc')   


## Load necessary parameters
HH = ds['nersc_sar_primary'].values

u = ds['u10m_rotated'].values   # wind azimuth
v = ds['v10m_rotated'].values   # wind range
W_pow = np.sqrt(u**2 + v**2)

bt_6_9_h = ds['btemp_6_9h'].values
bt_36_5_h = ds['btemp_36_5h'].values

V = ds['tcwv'].values 
L = ds['tclw'].values
Ta = ds['t2m'].values

Ts=275.0
Ti=260.0
Ti_amsrv = Ti*np.ones(8)
Ti_amsrh = Ti*np.ones(8)
c_ice=0.0
e_icev=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
e_iceh=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])

idx_interest = np.array([1,9])
AMSR2_model = np.empty((W_pow.shape[0], W_pow.shape[1], 12))

RTM_base = amsr(0,0,0,np.mean(Ta),Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)

for i in range(W_pow.shape[0]):
    for j in range(W_pow.shape[1]):
        AMSR2_model[i,j,:] = amsr(V[i,j],W_pow[i,j],L[i,j],Ta[i,j],Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)

TB_corr = TB_correction(bt_6_9_h, AMSR2_model[:,:,idx_interest[0]], RTM_base[idx_interest[0]])
TB_corr2 = TB_correction(bt_36_5_h, AMSR2_model[:,:,idx_interest[1]], RTM_base[idx_interest[1]])

x = np.linspace(0,np.shape(W_pow)[1]-1,np.shape(W_pow)[1])
y = np.linspace(0,np.shape(W_pow)[0]-1,np.shape(W_pow)[0])
f_low = interpolate.interp2d(x, y, AMSR2_model[:,:,idx_interest[0]], kind='linear')
f_high = interpolate.interp2d(x, y, AMSR2_model[:,:,idx_interest[1]], kind='linear')
f_6_9 = interpolate.interp2d(x, y, bt_6_9_h, kind='linear')
f_36_5 = interpolate.interp2d(x, y, bt_36_5_h, kind='linear')


x = np.linspace(0,np.shape(W_pow)[1]-1,np.shape(HH)[1])
y = np.linspace(0,np.shape(W_pow)[0]-1,np.shape(HH)[0])
amsr2_low = f_low(x,y)
amsr2_high= f_high(x,y)
bt_6_9_h_reshape = f_6_9(x,y)
bt_36_5_h_reshape = f_36_5(x,y)

plt.imshow(amsr2_low)
plt.imshow(amsr2_high)

## Normalize values to work with the images
HH_norm = normalize(HH)
amsr2_low_norm = normalize(amsr2_low)
amsr2_high_norm = normalize(amsr2_high)


diff_low = normalize(bt_6_9_h_reshape) - amsr2_low_norm
diff_high = normalize(bt_36_5_h_reshape) - amsr2_high_norm

## Masking features trials
aux_low= HH_norm * diff_low
aux_high = HH_norm * diff_high

aux_subtract_low = HH_norm - diff_low
aux_subtract_high = HH_norm - diff_high

aux_add_low = HH_norm + diff_low
aux_add_high = HH_norm + diff_high



############ PLOTTING ############

fig, ax = plt.subplots()
pos = ax.imshow(HH_norm, cmap='RdBu_r')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='HH_norm scene {}'.format(scene))
plt.savefig('HH_norm_s{}.png'.format(scene), format='PNG', dpi=300)

plt.figure(figsize=(12, 4.5))
ax1 = plt.subplot(1, 2, 1)
plt.imshow(amsr2_low_norm, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax1.title.set_text('Low frequency model')
ax2 = plt.subplot(1, 2, 2)
plt.imshow(amsr2_high_norm, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax2.title.set_text('High frequency model')
plt.suptitle('AMSR2 MODEL outputs for scene {}'.format(scene))
plt.savefig('AMSR2_s{}.png'.format(scene), format='PNG', dpi=300)

plt.figure(figsize=(12, 4.5))
ax1 = plt.subplot(1, 2, 1)
plt.imshow(bt_6_9_h_reshape, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax1.title.set_text('Low frequency model')
ax2 = plt.subplot(1, 2, 2)
plt.imshow(bt_36_5_h_reshape, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax2.title.set_text('High frequency model')
plt.suptitle('AMSR2 measurements for scene {}'.format(scene))
plt.savefig('AMSR2_measurements_s{}.png'.format(scene), format='PNG', dpi=300)


plt.figure(figsize=(12, 4.5))
ax1 = plt.subplot(1, 2, 1)
plt.imshow(aux_low, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax1.title.set_text('AMSR2 low freq. product')
ax2 = plt.subplot(1, 2, 2)
plt.imshow(aux_high, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax2.title.set_text('AMSR2 high freq. model product')
plt.suptitle('HH_norm * AMSR2 MODEL / scene {}'.format(scene))
plt.savefig('AMSR2_product_s{}.png'.format(scene), format='PNG', dpi=300)


plt.figure(figsize=(12, 4.5))
ax1 = plt.subplot(1, 2, 1)
plt.imshow(aux_subtract_low, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax1.title.set_text('AMSR2 LOW freq. subtract')
ax2 = plt.subplot(1, 2, 2)
plt.imshow(aux_subtract_high, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax2.title.set_text('AMSR2 high freq. subtract')
plt.suptitle('HH_norm - AMSR2 MODEL / scene {}'.format(scene))
plt.savefig('AMSR2_subtract_s{}.png'.format(scene), format='PNG', dpi=300)


plt.figure(figsize=(12, 4.5))
ax1 = plt.subplot(1, 2, 1)
plt.imshow(aux_add_low, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax1.title.set_text('AMSR2 LOW freq. add')
ax2 = plt.subplot(1, 2, 2)
plt.imshow(aux_add_high, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax2.title.set_text('AMSR2 high freq. add')
plt.suptitle('HH_norm + AMSR2 MODEL / scene {}'.format(scene))
plt.savefig('AMSR2_add_s{}.png'.format(scene), format='PNG', dpi=300)

'''
## Try a binary classifier OW - SI
th_05, th_95 = np.percentile(aux_subtract, np.array([5,95]))
ind_0 = np.where(aux_subtract >= th_95)
ind_1 = np.where(aux_subtract <= th_05)

quak = aux_subtract * 0
quak[ind_0] = 1
quak[ind_1] = 1


fig, ax = plt.subplots()
pos = ax.imshow(quak, cmap='RdBu_r')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='CMOD5N LOG10 scatter model')
'''

## Idea: Different sea ice types defined by different quantiles


