# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 11:28:05 2022

@author: Pau

Description: PCA for AMSR2 data for both H and V polarizations. The objective 
is to see if a mask can be better extracted by combining informations in 
multiple bands and polarizations.
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


def PCA_custom(Xflat, transposeS):
    meanV = np.mean(Xflat, axis=0)
    stdV = np.std(Xflat, axis=0)
    Xst = (Xflat-meanV) / stdV

    cov = np.cov(np.transpose(Xst))
    S, V, D = np.linalg.svd(cov)
    if transposeS:
        S = np.transpose(S)

    score = 100 * ( V / np.sum(V) )
    ratio = np.empty((np.shape(Xflat)[1], np.shape(Xflat)[1]))
    for i in range(5):
        for j in range(5):
            ratio[i, j] = S[i, j] * np.sqrt(V[j]) / stdV[i]


    cov_PCs = np.cov(S)     # Almost not correlated 
    Xst_PC = np.dot(Xst, S)
    
    return Xst_PC, cov, score, S, ratio
    

scene = 4   
## Get scene
# Scene 04
if scene ==4:
    ds = xr.open_dataset('S1A_EW_GRDM_1SDH_20190507T101118_20190507T101218_027120_030E8D_4E4F_icechart_cis_SGRDINFLD_20190507T1011Z_pl_a.nc')   
elif scene == 14:
    ds = xr.open_dataset('S1B_EW_GRDM_1SDH_20180424T212356_20180424T212505_010631_013669_E657_icechart_cis_SGRDINFLD_20180424T2121Z_pl_a.nc')   


## Load necessary parameters
HH = ds['nersc_sar_primary'].values


# RADIOMETRIC data from AMSR2
bt_6_9_h = ds['btemp_6_9h'].values
bt_10_7_h = ds['btemp_10_7h'].values
bt_18_7_h = ds['btemp_18_7h'].values
bt_23_8_h = ds['btemp_23_8h'].values
bt_36_5_h = ds['btemp_36_5h'].values
bt_89_0_h = ds['btemp_89_0h'].values

bt_6_9_v = ds['btemp_6_9v'].values
bt_10_7_v = ds['btemp_10_7v'].values
bt_18_7_v = ds['btemp_18_7v'].values
bt_23_8_v = ds['btemp_23_8v'].values
bt_36_5_v = ds['btemp_36_5v'].values
bt_89_0_v = ds['btemp_89_0v'].values

orig_size = np.shape(bt_6_9_h)


# RADIOMETRIC MODEL FOR AMSR2 USING ERAS05 DATASET
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
e_icev=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
e_iceh=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])

idx_interest = np.array([1,9])
AMSR2_model = np.empty((W_pow.shape[0], W_pow.shape[1], 12))

RTM_base = amsr(0,0,0,np.mean(Ta),Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)

for i in range(W_pow.shape[0]):
    for j in range(W_pow.shape[1]):
        AMSR2_model[i,j,:] = amsr(V[i,j],W_pow[i,j],L[i,j],Ta[i,j],Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)

#TB_corr = TB_correction(bt_6_9_h, AMSR2_model[:,:,idx_interest[0]], RTM_base[idx_interest[0]])
#TB_corr2 = TB_correction(bt_36_5_h, AMSR2_model[:,:,idx_interest[1]], RTM_base[idx_interest[1]])

transposeS = False

# PCA for AMSR2 measurements
TB_meas = np.column_stack((bt_6_9_v.flatten(), bt_6_9_h.flatten(), 
                           bt_10_7_v.flatten(), bt_10_7_h.flatten(), 
                           bt_18_7_v.flatten(), bt_18_7_h.flatten(), 
                           bt_23_8_v.flatten(), bt_23_8_h.flatten(), 
                           bt_36_5_v.flatten(), bt_36_5_h.flatten(), 
                           bt_89_0_h.flatten(), bt_89_0_v.flatten()))
Xst_PCA, cov, score, S, ratio = PCA_custom(TB_meas, transposeS)

# PCA for AMSR2 model output
TB_model = np.column_stack((AMSR2_model[:,:,0].flatten(), AMSR2_model[:,:,1].flatten(),
                           AMSR2_model[:,:,2].flatten(), AMSR2_model[:,:,3].flatten(), 
                           AMSR2_model[:,:,4].flatten(), AMSR2_model[:,:,5].flatten(), 
                           AMSR2_model[:,:,6].flatten(), AMSR2_model[:,:,7].flatten(), 
                           AMSR2_model[:,:,8].flatten(), AMSR2_model[:,:,9].flatten(), 
                           AMSR2_model[:,:,10].flatten(), AMSR2_model[:,:,11].flatten()))
Xst_PCA_m, cov_m, score_m, S_m, ratio_m = PCA_custom(TB_model, transposeS)


# PCA for AMSR2 meas & model output
TB_both = np.column_stack((bt_6_9_v.flatten(), bt_6_9_h.flatten(), 
                           bt_10_7_v.flatten(), bt_10_7_h.flatten(), 
                           bt_18_7_v.flatten(), bt_18_7_h.flatten(), 
                           bt_23_8_v.flatten(), bt_23_8_h.flatten(), 
                           bt_36_5_v.flatten(), bt_36_5_h.flatten(), 
                           bt_89_0_h.flatten(), bt_89_0_v.flatten(),
                           AMSR2_model[:,:,0].flatten(), AMSR2_model[:,:,1].flatten(),
                           AMSR2_model[:,:,2].flatten(), AMSR2_model[:,:,3].flatten(), 
                           AMSR2_model[:,:,4].flatten(), AMSR2_model[:,:,5].flatten(), 
                           AMSR2_model[:,:,6].flatten(), AMSR2_model[:,:,7].flatten(), 
                           AMSR2_model[:,:,8].flatten(), AMSR2_model[:,:,9].flatten(), 
                           AMSR2_model[:,:,10].flatten(), AMSR2_model[:,:,11].flatten()))
Xst_PCA_both, cov_both, score_both, S_both, ratio_both = PCA_custom(TB_both, transposeS)

## False color composite image

#img_false = np.dstack((imgf_PC[:,0].reshape(orig_size), imgf_PC[:,1].reshape(orig_size), imgf_PC[:,2].reshape(orig_size)))
Xst_false_meas = np.dstack((Xst_PCA[:,0].reshape(orig_size), Xst_PCA[:,1].reshape(orig_size), Xst_PCA[:,2].reshape(orig_size)))
Xst_false_model = np.dstack((Xst_PCA_m[:,0].reshape(orig_size), Xst_PCA_m[:,1].reshape(orig_size), Xst_PCA_m[:,2].reshape(orig_size)))
Xst_false_both = np.dstack((Xst_PCA_both[:,0].reshape(orig_size), Xst_PCA_both[:,1].reshape(orig_size), Xst_PCA_both[:,2].reshape(orig_size)))


f = plt.figure()
plt.imshow(Xst_false_meas)
plt.savefig('False_composite_3_PCs_MSR2_meas.png')
plt.show()

f = plt.figure()
plt.imshow(Xst_false_model)
plt.savefig('False_composite_3_PCs_MSR2_model.png')
plt.show()

f = plt.figure()
plt.imshow(Xst_false_both)
plt.savefig('False_composite_3_PCs_MSR2_both.png')
plt.show()


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


