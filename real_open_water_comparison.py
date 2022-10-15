# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:05:41 2022

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

'''
# Select open water images dataset
file_list = []
i = 0
for file in (glob.glob("*.nc")):
    
    if ((i == 4) or (i == 14)):
        print(file)
        ds = xr.open_dataset(file)
        file_list.append(ds);
    i = i+1
 '''

def normalize(X):
    
    X_max = np.nanmax(X)
    X_min = np.nanmin(X)

    X_out = np.divide((X - X_min), (X_max - X_min))
    
    return np.nan_to_num(X_out)



# Scene 04
ds = xr.open_dataset('S1A_EW_GRDM_1SDH_20190507T101118_20190507T101218_027120_030E8D_4E4F_icechart_cis_SGRDINFLD_20190507T1011Z_pl_a.nc')   
# Scene 14
#ds = xr.open_dataset('S1B_EW_GRDM_1SDH_20180424T212356_20180424T212505_010631_013669_E657_icechart_cis_SGRDINFLD_20180424T2121Z_pl_a.nc')   

# Comparison between AMSR2 measurements and model
HH = ds['nersc_sar_primary'].values
dm = ds['distance_map'].values
poly_icechart = ds['polygon_icechart'].values  # Polygon ice chart

bt_6_9_h = ds['btemp_6_9h'].values   # RADIOMETRIC data from AMSR2
bt_10_7_h = ds['btemp_10_7h'].values   # RADIOMETRIC data from AMSR2
bt_18_7_h = ds['btemp_18_7h'].values
bt_23_8_h = ds['btemp_23_8h'].values
bt_36_5_h = ds['btemp_36_5h'].values
bt_89_0_h = ds['btemp_89_0h'].values   # RADIOMETRIC data from AMSR2

bt_6_9_v = ds['btemp_6_9v'].values   # RADIOMETRIC data from AMSR2
bt_10_7_v = ds['btemp_10_7v'].values   # RADIOMETRIC data from AMSR2
bt_18_7_v = ds['btemp_18_7v'].values
bt_23_8_v = ds['btemp_23_8v'].values
bt_36_5_v = ds['btemp_36_5v'].values
bt_89_0_v = ds['btemp_89_0v'].values   # RADIOMETRIC data from AMSR2



u = ds['u10m_rotated'].values   # wind azimuth
v = ds['v10m_rotated'].values   # wind range
W_pow = np.sqrt(u**2 + v**2)
phi = np.arccos(u/W_pow)*180/np.pi

### AMSR2 MODELLING GIVEN PHYSICAL PARAM. ###
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

# Iterate pixel x pixel
n_freq = 12
amsr_multi_freq = np.empty((np.shape(V)[0], np.shape(V)[1], n_freq))

for i in range(np.shape(V)[0]):
    for j in range(np.shape(V)[1]):
       amsr_multi_freq[i,j,:] = amsr(V[i,j],W_pow[i,j],L[i,j],Ta[i,j],Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)



### COMPARISSON
# 6.9 GHz Comparison
plt.figure(figsize=(12, 4.5))

ax1 = plt.subplot(1, 2, 1)
plt.imshow(bt_6_9_h, cmap='RdBu')
quak = plt.colorbar(extend='both')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
ax1.title.set_text('Measurements')

ax2 = plt.subplot(1, 2, 2)
plt.imshow(amsr_multi_freq[:,:,1], cmap='RdBu')
quak = plt.colorbar(extend='both')
ax2.title.set_text('Model')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
plt.suptitle('AMSR2 TB_H comparison at 6.9 GHz')
plt.savefig('S4_AMSR2_H_6,9GHz.png'.format(i), format='PNG', dpi=300)

# 10.7 GHz Comparison
plt.figure(figsize=(12, 4.5))

ax1 = plt.subplot(1, 2, 1)
plt.imshow(bt_10_7_h, cmap='RdBu')
quak = plt.colorbar(extend='both')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
ax1.title.set_text('Measurements')

ax2 = plt.subplot(1, 2, 2)
plt.imshow(amsr_multi_freq[:,:,3], cmap='RdBu')
quak = plt.colorbar(extend='both')
ax2.title.set_text('Model')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
plt.suptitle('AMSR2 TB_H comparison at 10.7 GHz')
plt.savefig('S4_AMSR2_H_10,7GHz.png'.format(i), format='PNG', dpi=300)

# 18.7 GHz Comparison
plt.figure(figsize=(12, 4.5))

ax1 = plt.subplot(1, 2, 1)
plt.imshow(bt_18_7_h, cmap='RdBu')
quak = plt.colorbar(extend='both')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
ax1.title.set_text('Measurements')

ax2 = plt.subplot(1, 2, 2)
plt.imshow(amsr_multi_freq[:,:,5], cmap='RdBu')
quak = plt.colorbar(extend='both')
ax2.title.set_text('Model')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
plt.suptitle('AMSR2 TB_H comparison at 18.7 GHz')
plt.savefig('S4_AMSR2_H_18,7GHz.png'.format(i), format='PNG', dpi=300)

# 23.89 GHz Comparison
plt.figure(figsize=(12, 4.5))

ax1 = plt.subplot(1, 2, 1)
plt.imshow(bt_23_8_h, cmap='RdBu')
quak = plt.colorbar(extend='both')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
ax1.title.set_text('Measurements')

ax2 = plt.subplot(1, 2, 2)
plt.imshow(amsr_multi_freq[:,:,7], cmap='RdBu')
quak = plt.colorbar(extend='both')
ax2.title.set_text('Model')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
plt.suptitle('AMSR2 TB_H comparison at 23.8 GHz')
plt.savefig('S4_AMSR2_H_23,8GHz.png'.format(i), format='PNG', dpi=300)

# 36.5 GHz Comparison
plt.figure(figsize=(12, 4.5))

ax1 = plt.subplot(1, 2, 1)
plt.imshow(bt_36_5_h, cmap='RdBu')
quak = plt.colorbar(extend='both')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
ax1.title.set_text('Measurements')

ax2 = plt.subplot(1, 2, 2)
plt.imshow(amsr_multi_freq[:,:,9], cmap='RdBu')
quak = plt.colorbar(extend='both')
ax2.title.set_text('Model')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
plt.suptitle('AMSR2 TB_H comparison at 36.5 GHz')
plt.savefig('S4_AMSR2_H_36,5GHz.png'.format(i), format='PNG', dpi=300)

# 89.0 GHz Comparison
plt.figure(figsize=(12, 4.5))

ax1 = plt.subplot(1, 2, 1)
plt.imshow(bt_89_0_h, cmap='RdBu')
quak = plt.colorbar(extend='both')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
ax1.title.set_text('Measurements')

ax2 = plt.subplot(1, 2, 2)
plt.imshow(amsr_multi_freq[:,:,11], cmap='RdBu')
quak = plt.colorbar(extend='both')
ax2.title.set_text('Model')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
plt.suptitle('AMSR2 TB_H comparison at 89.0 GHz')
plt.savefig('S4_AMSR2_H_89,0GHz.png'.format(i), format='PNG', dpi=300)


############################################################


# HH pol
fig, ax = plt.subplots()
pos = ax.imshow(HH, cmap='RdBu')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='HH pol scatter')
plt.savefig('S4_AMSR2_HH.png'.format(i), format='PNG', dpi=300)

## Distance Map
fig, ax = plt.subplots()
pos = ax.imshow(dm, cmap='RdBu')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='Distance map')
plt.savefig('S4_AMSR2_dm.png'.format(i), format='PNG', dpi=300)

## Wind speed
fig, ax = plt.subplots()
pos = ax.imshow(W_pow, cmap='RdBu')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='Wind speed [m/s]')
plt.savefig('S4_AMSR2_W.png'.format(i), format='PNG', dpi=300)

## L
fig, ax = plt.subplots()
pos = ax.imshow(L, cmap='RdBu')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='Columnar cloud liquid water [mm]')
plt.savefig('S4_AMSR2_L.png'.format(i), format='PNG', dpi=300)

## V
fig, ax = plt.subplots()
pos = ax.imshow(V, cmap='RdBu')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='Columnar water vapor [mm]')
plt.savefig('S4_AMSR2_V.png'.format(i), format='PNG', dpi=300)

## phi
fig, ax = plt.subplots()
pos = ax.imshow(phi, cmap='RdBu')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='Wind corrected $\phi$ ')
plt.savefig('S4_AMSR2_phi.png'.format(i), format='PNG', dpi=300)


## CMOD5N MODEL COMPARISON - SCATTERING

u = ds['u10m_rotated'].values   # wind azimuth
v = ds['v10m_rotated'].values   # wind range
W_pow = np.sqrt(u**2 + v**2)
phi = np.arccos(u/W_pow)*180/np.pi

incidence = ds['sar_grid_incidenceangle'].values
inc_2D = incidence.reshape(21,21)
aux = inc_2D[:21,:]
# Reshaping SAR coord. to 2km grid --> Interpolate data tp 2km grid size
x = np.linspace(0,np.shape(u)[1]-1,21)
y = np.linspace(0,np.shape(u)[0],21)
f = interpolate.interp2d(x, y, aux, kind='linear')

x = np.linspace(0, np.shape(u)[1]-1, np.shape(u)[1])
y = np.linspace(0, np.shape(u)[0]-1, np.shape(u)[0])
theta_reshaped = f(x,y)

# Simulate cmod5n model values for Scatter on Open Water wind
cmod5n_values = cmod5n_forward(W_pow,phi,theta_reshaped)

# CMOD5N pol
fig, ax = plt.subplots()
pos = ax.imshow(cmod5n_values, cmap='RdBu')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='CMOD5N scatter model')
plt.savefig('S4_CMOD5N.png', format='PNG', dpi=300)

# CMOD5N pol
fig, ax = plt.subplots()
pos = ax.imshow(10*np.log10(cmod5n_values), cmap='RdBu')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='CMOD5N LOG10 scatter model')
plt.savefig('S4_CMOD5N_LOG10.png', format='PNG', dpi=300)



######################## Feautre masking ########################
## Feature masking with CMOD5N model

# HH pol
fig, ax = plt.subplots()
pos = ax.imshow(HH, cmap='RdBu_r')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='CMOD5N LOG10 scatter model')
#plt.savefig('S4_HH.png', format='PNG', dpi=300)

# CMOD5N pol
fig, ax = plt.subplots()
pos = ax.imshow(cmod5n_values, cmap='RdBu_r')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='CMOD5N scatter model')
#plt.savefig('S4_CMOD5N.png', format='PNG', dpi=300)

# CMOD5N log10 pol
fig, ax = plt.subplots()
pos = ax.imshow(10*np.log10(cmod5n_values), cmap='RdBu_r')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='CMOD5N LOG10 scatter model')
#plt.savefig('S4_CMOD5N_LOG10.png', format='PNG', dpi=300)


# Normalize functions
#re-scale cmod5n size with SAR image size

u = ds['u10m_rotated'].values   # wind azimuth
v = ds['v10m_rotated'].values   # wind range
W_pow = np.sqrt(u**2 + v**2)
phi = np.arccos(u/W_pow)*180/np.pi

incidence = ds['sar_grid_incidenceangle'].values
inc_2D = incidence.reshape(21,21)
aux = inc_2D[:21,:]

# Reshaping SAR coord. to 2km grid --> Interpolate data tp 2km grid size
x = np.linspace(0,np.shape(HH)[1]-1,21)
y = np.linspace(0,np.shape(HH)[0],21)
f = interpolate.interp2d(x, y, aux, kind='linear')

x = np.linspace(0, np.shape(HH)[1]-1, np.shape(HH)[1])
y = np.linspace(0, np.shape(HH)[0]-1, np.shape(HH)[0])
theta_reshaped = f(x,y)


x = np.linspace(0,np.shape(HH)[1]-1,np.shape(W_pow)[1])
y = np.linspace(0,np.shape(HH)[0],np.shape(W_pow)[0])
f_W = interpolate.interp2d(x, y, W_pow, kind='linear')
f_phi = interpolate.interp2d(x, y, phi, kind='linear')

x = np.linspace(0, np.shape(HH)[1]-1, np.shape(HH)[1])
y = np.linspace(0, np.shape(HH)[0]-1, np.shape(HH)[0])
W_pow_reshaped = f_W(x,y)
phi_reshaped = f_phi(x,y)

# Simulate cmod5n model values for Scatter on Open Water wind
cmod5n_values = cmod5n_forward(W_pow_reshaped,phi_reshaped,theta_reshaped)

HH_norm = normalize(HH)
cmod5n_norm = normalize(cmod5n_values)
cmod5n_log_norm = normalize(10*np.log10(cmod5n_values))

aux = HH_norm * cmod5n_norm
aux_log = np.multiply(HH_norm, cmod5n_log_norm)

aux_subtract = HH_norm - cmod5n_log_norm

fig, ax = plt.subplots()
pos = ax.imshow(aux, cmap='RdBu_r')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='CMOD5N LOG10 scatter model')

fig, ax = plt.subplots()
pos = ax.imshow(aux_log, cmap='RdBu_r')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='CMOD5N LOG10 scatter model')

fig, ax = plt.subplots()
pos = ax.imshow(aux_subtract, cmap='RdBu_r')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='CMOD5N LOG10 scatter model')





### COMPARISSON
# 6.9 GHz Comparison
plt.figure(figsize=(12, 4.5))

ax1 = plt.subplot(1, 2, 1)
plt.imshow(bt_6_9_h, cmap='RdBu')
quak = plt.colorbar(extend='both')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
ax1.title.set_text('Measurements')

ax2 = plt.subplot(1, 2, 2)
plt.imshow(amsr_multi_freq[:,:,1], cmap='RdBu')
quak = plt.colorbar(extend='both')
ax2.title.set_text('Model')
quak.set_label('TB_H [K]', rotation=270, labelpad=20)
plt.suptitle('AMSR2 TB_H comparison at 6.9 GHz')
plt.savefig('S4_AMSR2_H_6,9GHz.png'.format(i), format='PNG', dpi=300)





















