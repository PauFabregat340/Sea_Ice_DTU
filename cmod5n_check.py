# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:49:10 2022

@author: Pau
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from cmod5n import cmod5n_forward

wind_speed = 10.0 #m/s

inc_max = 45.0
inc_min = 20.0
inc_vec = np.linspace(inc_min, inc_max,5)

phi_min = 0
phi_max = 360
phi_vec = np.linspace(phi_min, phi_max, 721) # Step of 0.5º

cmod5n_results = np.empty((len(phi_vec),len(inc_vec)))

fig, ax = plt.subplots()

for i, inc in enumerate(inc_vec):
    cmod5n_results[:,i] = cmod5n_forward(wind_speed * np.ones(len(phi_vec)), phi_vec, inc * np.ones(len(phi_vec)))
    ax.plot(phi_vec, cmod5n_results[:,i], label = 'Inc = {}º'.format(inc))
    #ax.plot(phi_vec, 10*np.log10(cmod5n_results[:,i]), label = 'Inc = {}º'.format(inc))
ax.set(xlabel = '$\Phi$ [º]', ylabel='Normalized $\sigma$', title='CMOD5n backscatter model')
ax.legend(loc='best')
fig.savefig('cmod5n_phi.png', format='PNG', dpi=300)


fig, ax = plt.subplots()
ax.plot(inc_vec, 10*np.log10(cmod5n_results[0,:]), label = '$\Phi$ = 0º')
ax.plot(inc_vec, 10*np.log10(cmod5n_results[90,:]), label = '$\Phi$ = 45º')
ax.plot(inc_vec, 10*np.log10(cmod5n_results[180,:]), label = '$\Phi$ = 90º')
ax.plot(inc_vec, 10*np.log10(cmod5n_results[270,:]), label = '$\Phi$ = 135º')
ax.plot(inc_vec, 10*np.log10(cmod5n_results[360,:]), label = '$\Phi$ = 180º')
ax.set(xlabel = 'Incidence angle $\Theta$ [º]', ylabel='Log10 Normalized $\sigma$', title='CMOD5n $\sigma$ decrease for $\Theta$')
ax.legend(loc='best')
fig.savefig('cmod5n_theta.png', format='PNG', dpi=300)

rate = cmod5n_results[360,:]/cmod5n_results[360,0]

aux = np.max(cmod5n_results, axis=0)
rate2 = np.divide(cmod5n_results,aux)
fig, ax = plt.subplots()
ax.plot(phi_vec, rate2)
ax.set(xlabel = '$\Phi$ [º]', ylabel='2 x Normalized $\sigma$', title='Backscatter decrease rate')
ax.legend(loc='best')
fig.savefig('cmod5n_rate.png', format='PNG', dpi=300)
