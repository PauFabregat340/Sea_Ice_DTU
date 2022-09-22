# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:53:18 2022

@author: Pau
"""

import ftplib
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from cmod5n import cmod5n_forward
from amsrrtm2 import amsr
import math
import cmath

## Download data form dataset via FTP
host = 'ftp.dmi.dk'
usr = 'ASIP'
pw = 'TEWVK.wdcvx'
dataset_dir = 'AI4Arctic_challenge_data'

with ftplib.FTP(host=host, user=usr, passwd=pw) as ftp:
    list_of_netcdfs = [file for file in ftp.nlst(dataset_dir) if file.endswith('.nc')]
    test_file = list_of_netcdfs[2]
    with open(os.path.join(os.getcwd(), os.path.basename(test_file)), 'wb') as f:
        ftp.retrbinary("RETR " + test_file, f.write)
ds = xr.open_dataset(os.path.basename(test_file))

### GET PARAMETERS FROM THE DATASET ###
# Get Images in HH and HV Polarization
HH = ds['nersc_sar_primary'].values
HV = ds['nersc_sar_secondary'].values

# SAR parameters
incidence = ds['sar_grid_incidenceangle'].values
inc_2D = incidence.reshape(21,21)
height = ds['sar_grid_height'].values
lat = ds['sar_grid_latitude'].values
lon = ds['sar_grid_longitude'].values

# SAR GRID params
grid_line = ds['sar_grid_line'].values
grid_points = ds['sar_grid_points'].values
grid_sample = ds['sar_grid_sample'].values
lines = ds['sar_lines'].values
samples = ds['sar_samples'].values

# Distance map
dm = ds['distance_map'].values

# Polygon ice chart
poly_icechart = ds['polygon_icechart'].values
poly_codes = ds['polygon_codes'].values

# RADIOMETRIC data from AMSR2
bt_10_7_h = ds['btemp_10_7h'].values
bt_18_7_h = ds['btemp_18_7h'].values
bt_23_8_h = ds['btemp_23_8h'].values
bt_36_5_h = ds['btemp_36_5h'].values

# WIND information from ERAS5 dataset
u = ds['u10m_rotated'].values   #azimuth
v = ds['v10m_rotated'].values   #range

### CMOD5N function with FTP params. ###
# phi in [deg] angle between azimuth and wind direction (= D - AZM)
# theta in [deg] incidence angle

# Obtain wind speed/power, and angle w.r.t Azimuth
W_pow = np.sqrt(u**2 + v**2)
phi = -np.arcsin(v/W_pow)*180/np.pi

# Reshaping SAR coord. to 2km grid --> Interpolate data tp 2km grid size
x = np.linspace(0,np.shape(u)[1]-1,21)
y = np.linspace(0,np.shape(u)[0],21)
f = interpolate.interp2d(x, y, inc_2D, kind='linear')

x = np.linspace(0, np.shape(u)[1]-1, np.shape(u)[1])
y = np.linspace(0, np.shape(u)[0]-1, np.shape(u)[0])
theta_reshaped = f(x,y)

# Simulate cmod5n model values for Scatter on Open Water wind
cmod5n_values = cmod5n_forward(W_pow,phi,theta_reshaped)



### AMSR2 MODELLING GIVEN PHYSICAL PARAM. ###
'''
Tb=f(V,W,L,Ts,Ti,c_ice)
V: columnar water vapor [mm]
W: windspeed over water [m/s]
L: columnar cloud liquid water [mm]
Ts: sea surface temperature [K]
Ti: ice surface temperature [K]
c_ice: ice concentration [0-1]
'''
# Get params. form dataset
V = ds['tcwv'].values 
L = ds['tclw'].values
Ta = ds['t2m'].values

# Introduce params. manually
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
       amsr_multi_freq[i,j,:] = amsr(V[i,j],W_pow[i,j],L[i,j],Ta[i,j],Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,theta_reshaped[i,j])

fig, ax = plt.subplots()
pos = ax.imshow(amsr_multi_freq[:,:,2])
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
   

### Plotting results ###

# SAR image in HH
fig, ax = plt.subplots()
pos = ax.imshow(dm, origin='lower')
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()



# Winds
# U wind
fig, ax = plt.subplots()
pos = ax.imshow(u)
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='U wind speed (Azimuth)')

# V wind
fig, ax = plt.subplots()
pos = ax.imshow(v)
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='V wind speed (RANGE)')

# W_pow wind
fig, ax = plt.subplots()
pos = ax.imshow(W_pow)
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='W wind speed (Azimuth)')

# Cmond5n estimates wind
fig, ax = plt.subplots()
pos = ax.imshow(cmod5n_values)
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
ax.set(title='CMOD5N wind speed (Azimuth)')

    
## Example of good plot
dm = ds['distance_map'].values
fig, ax = plt.subplots()
pos = ax.imshow(dm, extent=[lon.min(), lon.max(), lat.min(),lat.max()])#, origin="lower")
ax.set(xlabel='Longitude [º]', ylabel='Latitude [º]',
       title='Distance map from coastline ( 0-300+ km - 0-41)')
#ax.grid()
fig.savefig("Distance_map.png")
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
plt.show()


## Example of Quick visualization Plot
fig, ax = plt.subplots()
pos = ax.imshow(dm)
cbar = fig.colorbar(pos, ax=ax, extend='both')
cbar.minorticks_on()
   
