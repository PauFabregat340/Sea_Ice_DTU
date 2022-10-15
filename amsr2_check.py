# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:53:26 2022

@author: Pau
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from amsrrtm2 import amsr


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
W = 10 # m/s
V = 17.5 # mm
L = 0.3 # mm
Ta = 275

# Introduce params. manually
Ts=275.0
Ti=260.0
Ti_amsrv = Ti*np.ones(8)
Ti_amsrh = Ti*np.ones(8)
c_ice=0.0
e_icev=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
e_iceh=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])


### TESTS ###

# Iterate pixel x pixel
n_freq = 12
amsr_multi_freq = np.empty((3, n_freq))
frequencies=np.array([6.93, 10.65, 18.70, 23.80, 36.50, 89.00])
horizontal = np.array([1,3,5,7,9,11])
vertical = np.array([0,2,4,6,8,10])

# W: windspeed over water [m/s]
amsr_multi_freq[0,:] = amsr(V,1,L,Ta,Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)
amsr_multi_freq[1,:] = amsr(V,5,L,Ta,Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)
amsr_multi_freq[2,:] = amsr(V,9,L,Ta,Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)

fig, ax = plt.subplots()
ax.plot(frequencies, amsr_multi_freq[0, horizontal], 'b--', label = 'TH, W = 1 m/s')
ax.plot(frequencies, amsr_multi_freq[0, vertical], 'b-',  label = 'TV, W = 1 m/s')
ax.plot(frequencies, amsr_multi_freq[1, horizontal], 'r--',  label = 'TH, W = 5 m/s')
ax.plot(frequencies, amsr_multi_freq[1, vertical], 'r-',  label = 'TV, W = 5 m/s')
ax.plot(frequencies, amsr_multi_freq[2, horizontal], 'g--',  label = 'TH, W = 9 m/s')
ax.plot(frequencies, amsr_multi_freq[2, vertical], 'g-',  label = 'TV, W = 9 m/s')
ax.set(xlabel = 'Frequency [GHz]', ylabel='TB [K]', title='AMSR2 model: W sensitivity')
ax.legend(loc='best')
fig.savefig('AMSR_V.png', format='PNG', dpi=300)


# V: columnar water vapor [mm]
amsr_multi_freq[0,:] = amsr(11,W,L,Ta,Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)
amsr_multi_freq[1,:] = amsr(16,W,L,Ta,Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)
amsr_multi_freq[2,:] = amsr(21,W,L,Ta,Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)

fig, ax = plt.subplots()
ax.plot(frequencies, amsr_multi_freq[0, horizontal], 'b--', label = 'TH, V = 11 mm')
ax.plot(frequencies, amsr_multi_freq[0, vertical], 'b-',  label = 'TV, V = 11 mm')
ax.plot(frequencies, amsr_multi_freq[1, horizontal], 'r--',  label = 'TH, V = 16 mm')
ax.plot(frequencies, amsr_multi_freq[1, vertical], 'r-',  label = 'TV, V = 16 mm')
ax.plot(frequencies, amsr_multi_freq[2, horizontal], 'g--',  label = 'TH, V = 21 mm')
ax.plot(frequencies, amsr_multi_freq[2, vertical], 'g-',  label = 'TV, V = 21 mm')
ax.set(xlabel = 'Frequency [GHz]', ylabel='TB [K]', title='AMSR2 model: V sensitivity')
ax.legend(loc='best')
fig.savefig('AMSR_W.png', format='PNG', dpi=300)


# L: columnar cloud liquid water [mm]
amsr_multi_freq[0,:] = amsr(V,W,0.01,Ta,Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)
amsr_multi_freq[1,:] = amsr(V,W,0.3,Ta,Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)
amsr_multi_freq[2,:] = amsr(V,W,0.8,Ta,Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)

fig, ax = plt.subplots()
ax.plot(frequencies, amsr_multi_freq[0, horizontal], 'b--', label = 'TH, L = 0.01 mm')
ax.plot(frequencies, amsr_multi_freq[0, vertical], 'b-',  label = 'TV, L = 0.01 mm')
ax.plot(frequencies, amsr_multi_freq[1, horizontal], 'r--',  label = 'TH, L = 0.3 mm')
ax.plot(frequencies, amsr_multi_freq[1, vertical], 'r-',  label = 'TV, L = 0.3 mm')
ax.plot(frequencies, amsr_multi_freq[2, horizontal], 'g--',  label = 'TH, L = 0.8 mm')
ax.plot(frequencies, amsr_multi_freq[2, vertical], 'g-',  label = 'TV, L = 0.8 mm')
ax.set(xlabel = 'Frequency [GHz]', ylabel='TB [K]', title='AMSR2 model: L sensitivity')
ax.legend(loc='best')
fig.savefig('AMSR_L.png', format='PNG', dpi=300)


# Ta: Air 2m height temperature [K]
amsr_multi_freq[0,:] = amsr(V,W,L,273,Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)
amsr_multi_freq[1,:] = amsr(V,W,L,276,Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)
amsr_multi_freq[2,:] = amsr(V,W,L,279,Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)

fig, ax = plt.subplots()
ax.plot(frequencies, amsr_multi_freq[0, horizontal], 'b--', label = 'TH, Ta = 273 K')
ax.plot(frequencies, amsr_multi_freq[0, vertical], 'b-',  label = 'TV, Ta = 273 K')
ax.plot(frequencies, amsr_multi_freq[1, horizontal], 'r--',  label = 'TH, Ta = 276 K')
ax.plot(frequencies, amsr_multi_freq[1, vertical], 'r-',  label = 'TV, Ta = 276 K')
ax.plot(frequencies, amsr_multi_freq[2, horizontal], 'g--',  label = 'TH, Ta = 279 K')
ax.plot(frequencies, amsr_multi_freq[2, vertical], 'g-',  label = 'TV, Ta = 279 K')
ax.set(xlabel = 'Frequency [GHz]', ylabel='TB [K]', title='AMSR2 model: Ta sensitivity')
ax.legend(loc='best')
fig.savefig('AMSR_Ta.png', format='PNG', dpi=300)


# Ts: sea surface temperature [K]
amsr_multi_freq[0,:] = amsr(V,W,L,Ta,273,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)
amsr_multi_freq[1,:] = amsr(V,W,L,Ta,275,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)
amsr_multi_freq[2,:] = amsr(V,W,L,Ta,277,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)

fig, ax = plt.subplots()
ax.plot(frequencies, amsr_multi_freq[0, horizontal], 'b--', label = 'TH, Ts = 273 K')
ax.plot(frequencies, amsr_multi_freq[0, vertical], 'b-',  label = 'TV, Ts = 273 K')
ax.plot(frequencies, amsr_multi_freq[1, horizontal], 'r--',  label = 'TH, Ts = 275 K')
ax.plot(frequencies, amsr_multi_freq[1, vertical], 'r-',  label = 'TV, Ts = 275 K')
ax.plot(frequencies, amsr_multi_freq[2, horizontal], 'g--',  label = 'TH, Ts = 277 K')
ax.plot(frequencies, amsr_multi_freq[2, vertical], 'g-',  label = 'TV, Ts = 277 K')
ax.set(xlabel = 'Frequency [GHz]', ylabel='TB [K]', title='AMSR2 model: Ts sensitivity')
ax.legend(loc='best')
fig.savefig('AMSR_Ts.png', format='PNG', dpi=300)




