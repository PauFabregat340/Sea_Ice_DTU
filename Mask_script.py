# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:29:26 2022

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


'''
scene = 14   

if scene ==4:
    ds = xr.open_dataset(dataset_path + 'S1A_EW_GRDM_1SDH_20190507T101118_20190507T101218_027120_030E8D_4E4F_icechart_cis_SGRDINFLD_20190507T1011Z_pl_a.nc')   
elif scene == 14:
    ds = xr.open_dataset(dataset_path + 'S1B_EW_GRDM_1SDH_20190814T130029_20190814T130120_017582_021125_AAE4_icechart_cis_SGRDIMID_20190814T1258Z_pl_a.nc')   
'''


def main(ds, scene):
## AMSR2 mask
    HH = ds['nersc_sar_primary'].values
    dm = ds['distance_map'].values
    
    # RADIOMETRIC data from AMSR2
    bt_6_9_h = ds['btemp_6_9h'].values
    bt_10_7_h = ds['btemp_10_7h'].values
    bt_18_7_h = ds['btemp_18_7h'].values
    bt_23_8_h = ds['btemp_23_8h'].values
    bt_36_5_h = ds['btemp_36_5h'].values
    
    D_mead_TB = np.dstack((bt_6_9_h, bt_10_7_h, bt_18_7_h, bt_23_8_h, bt_36_5_h))
    orig_size = np.shape(bt_6_9_h)
    
    
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
    e_icev=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
    e_iceh=np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
    
    idx_interest = np.array([1,3,5,7,9])
    AMSR2_model = np.empty((W_pow.shape[0], W_pow.shape[1], 12))
    
    RTM_base = amsr(0,0,0,np.mean(Ta),Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)
    
    for i in range(W_pow.shape[0]):
        for j in range(W_pow.shape[1]):
            AMSR2_model[i,j,:] = amsr(V[i,j],W_pow[i,j],L[i,j],Ta[i,j],Ts,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh,55)
    
    TB_corr = TB_correction(D_mead_TB, AMSR2_model[:,:,idx_interest], RTM_base[idx_interest])
    '''
    TB_corr_ = np.dstack((normalize(TB_corr[:,:,0]), 
                               normalize(TB_corr[:,:,1]), 
                               normalize(TB_corr[:,:,2]), 
                               normalize(TB_corr[:,:,3]), 
                               normalize(TB_corr[:,:,4]))) 
    '''
    TB_corr_ = np.dstack((TB_corr[:,:,0], 
                          TB_corr[:,:,1], 
                          TB_corr[:,:,2], 
                          TB_corr[:,:,3], 
                          TB_corr[:,:,4])) 
    mask_PC1 = np.sum(np.array([0.45, 0.45, 0.45, 0.45, 0.45]) * TB_corr_, axis=2)
    mask_PC2 = np.sum(np.array([-0.55, -0.32, 0.04, 0.17, 0.75]) * TB_corr_, axis=2)
    
    mask_PC1 = normalize(mask_PC1)
    mask_PC2 = normalize(mask_PC2)
    
    aux = mask_PC1 * mask_PC2
    plt.imshow(aux)
    
    aux2 = ratio(mask_PC1, mask_PC2)
    plt.imshow(aux2)
    
    
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
    f_amsr2 = interpolate.interp2d(x, y, aux2, kind='linear')
    
    x = np.linspace(0,np.shape(cmod5n_values)[1]-1,np.shape(HH)[1])
    y = np.linspace(0,np.shape(cmod5n_values)[0]-1,np.shape(HH)[0])
    amsr2_reshaped = f_amsr2(x,y) # [-1,1]
    cmod5n_reshape = f_cmod(x,y)
    cmod5n_reshape_log = 10*np.log10(cmod5n_reshape)
    
    #ratio_lin = ratio(HH, cmod5n_reshape)
    #ratio_log = ratio(HH, cmod5n_reshape_log)
    
    ## Normalize values to work with the images
    HH_norm = normalize(HH)
    cmod5n_norm = normalize(cmod5n_reshape)
    cmod5n_log_norm = normalize(cmod5n_reshape_log)
    
    ratio_lin_norm = ratio(HH_norm, cmod5n_norm)
    ratio_log_norm = ratio(HH_norm, cmod5n_log_norm)
    '''
    f = plt.figure()
    plt.imshow(ratio_lin, cmap='RdBu_r')
    
    f = plt.figure()
    plt.imshow(ratio_log, cmap='RdBu_r')
    
    f = plt.figure()
    plt.imshow(ratio_lin_norm, cmap='RdBu_r')
    
    f = plt.figure()
    plt.imshow(ratio_log_norm, cmap='RdBu_r')
    
    f = plt.figure()
    plt.imshow(HH, cmap='RdBu_r')
    '''
    #print(os.getcwd())
    
    os.chdir(images_path + 'onlyratio/')
    f = plt.figure()
    plt.imshow(ratio_log_norm, cmap='RdBu_r')
    plt.suptitle('MASK_ONLYRATIO_s{}'.format(scene))
    plt.savefig('MASK_ONLYRATIO_s{}.png'.format(scene), format='PNG', dpi=300)
    plt.close(f)

    os.chdir('../product')
    f = plt.figure()
    plt.imshow(ratio_log_norm * amsr2_reshaped, cmap='RdBu_r')
    plt.suptitle('MASK_PRODUCT_s{}'.format(scene))
    plt.savefig('MASK_PRODUCT_s{}.png'.format(scene), format='PNG', dpi=300)
    plt.close(f)

    os.chdir('../addition')
    f = plt.figure()
    plt.imshow(ratio_log_norm + amsr2_reshaped, cmap='RdBu_r')
    plt.suptitle('MASK_ADDITION_s{}'.format(scene))
    plt.savefig('MASK_ADDITION_s{}.png'.format(scene), format='PNG', dpi=300)
    plt.close(f)

    os.chdir('../substract')
    f = plt.figure()
    plt.imshow(-(ratio_log_norm - amsr2_reshaped), cmap='RdBu_r')
    plt.suptitle('MASK_SUBTRACTION_s{}'.format(scene))
    plt.savefig('MASK_SUBTRACTION_s{}.png'.format(scene), format='PNG', dpi=300)
    plt.close(f)

    os.chdir('../HH')
    plt.figure(figsize=(12, 4.5))
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(HH, cmap='RdBu_r')
    quak = plt.colorbar(extend='both')
    ax1.title.set_text('HH SAR image')
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(dm, cmap='RdBu_r')
    quak = plt.colorbar(extend='both')
    ax2.title.set_text('Distance map')
    plt.suptitle('Overview Scene {}'.format(scene))
    plt.savefig('Overwiew_s{}.png'.format(scene), format='PNG', dpi=300)
    plt.close(f)

    '''
    plt.savefig('AMSR2_s{}.png'.format(scene), format='PNG', dpi=300)
    f = plt.figure()
    plt.imshow(HH, cmap='RdBu_r')
    #plt.savefig('MASK_SUBTRACTION_s{}.png'.format(scene), format='PNG', dpi=300)
    plt.savefig('HH_s{}.png'.format(scene), format='PNG', dpi=300)
     
    os.chdir('../substract')
    f = plt.figure()
    plt.imshow(dm, cmap='RdBu_r')
    #plt.savefig('MASK_SUBTRACTION_s{}.png'.format(scene), format='PNG', dpi=300)
    plt.savefig('Distancemap_s{}.png'.format(scene), format='PNG', dpi=300)
   '''
    os.chdir('../../../Scripts')


if __name__ == '__main__':
    main(ds, scene)
