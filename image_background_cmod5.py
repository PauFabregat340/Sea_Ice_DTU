# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:34:10 2022

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


def ratio(sigma_meas, sigma_model):
    R = (sigma_meas - sigma_model) / (sigma_meas + sigma_model)
    return R
    
    
    
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
incidence = ds['sar_grid_incidenceangle'].values

W_pow = np.sqrt(u**2 + v**2)
phi = np.arccos(u/W_pow)*180/np.pi
inc_2D = incidence.reshape(21,21)

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

#plt.imshow(cmod5n_reshape)
cmod5n_reshape_log = 10*np.log10(cmod5n_reshape)
#plt.imshow(cmod5n_reshape_log)

## Normalize values to work with the images
HH_norm = normalize(HH)
cmod5n_norm = normalize(cmod5n_reshape)
cmod5n_log_norm = normalize(cmod5n_reshape_log)


## Masking features trials
aux = HH_norm * cmod5n_norm
aux_log = np.multiply(HH_norm, cmod5n_log_norm)

aux_subtract_lin = HH_norm - cmod5n_norm

aux_subtract_log = HH_norm - cmod5n_log_norm


fig, ax = plt.subplots()
pos = ax.imshow(HH_norm, cmap='RdBu_r')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='HH_norm scene {}'.format(scene))
plt.savefig('HH_norm_s{}.png'.format(scene), format='PNG', dpi=300)

plt.figure(figsize=(12, 4.5))
ax1 = plt.subplot(1, 2, 1)
plt.imshow(cmod5n_norm, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax1.title.set_text('Linear model')
ax2 = plt.subplot(1, 2, 2)
plt.imshow(cmod5n_log_norm, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax2.title.set_text('Log10 model')
plt.suptitle('CMOD5N MODEL outputs for scene {}'.format(scene))
plt.savefig('CMOD5N_s{}.png'.format(scene), format='PNG', dpi=300)


plt.figure(figsize=(12, 4.5))
ax1 = plt.subplot(1, 2, 1)
plt.imshow(aux, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax1.title.set_text('Linear model product')
ax2 = plt.subplot(1, 2, 2)
plt.imshow(aux_log, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax2.title.set_text('Log10 model product')
plt.suptitle('HH_norm * CMOD5N MODEL / scene {}'.format(scene))
plt.savefig('CMOD5N_product_s{}.png'.format(scene), format='PNG', dpi=300)


plt.figure(figsize=(12, 4.5))
ax1 = plt.subplot(1, 2, 1)
plt.imshow(aux_subtract_lin, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax1.title.set_text('Linear model subtract')
ax2 = plt.subplot(1, 2, 2)
plt.imshow(aux_subtract_log, cmap='RdBu_r')
quak = plt.colorbar(extend='both')
ax2.title.set_text('Log10 model subtract')
plt.suptitle('HH_norm - CMOD5N MODEL / scene {}'.format(scene))
plt.savefig('CMOD5N_subtract_s{}.png'.format(scene), format='PNG', dpi=300)



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


