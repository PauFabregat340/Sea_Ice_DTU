# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:34:32 2022

@author: Pau
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from amsrrtm2 import amsr
from cmod5n import cmod5n_forward
import ic_algs as ica


image = bt_36_5_h
size_crop = (49, 49)
auxx = crop_scene(image, size_crop)

def crop_scene(image, size_crop = (256, 256)):
    
    # Crop towards the center for the prefect size to perform multiple crops on the scene
    shape = np.shape(image)
    scenesx = int(np.trunc(shape[0]/size_crop[0]))
    scenesy = int(np.trunc(shape[1]/size_crop[1]))
    scenes_array = np.empty((size_crop[0], size_crop[1], int(scenesx*scenesy)))
    
    center_orig_image_x = (shape[0]/2) # - 1
    center_orig_image_y = (shape[1]/2) # - 1

    size1x = scenesx*size_crop[0]
    size1y= scenesy*size_crop[1]
    
    idx_x_upper = int(center_orig_image_x + np.ceil(size1x/2))
    idx_x_lower = int(center_orig_image_x - np.trunc(size1x/2))

    idx_y_upper = int(center_orig_image_y + np.ceil(size1y/2))
    idx_y_lower = int(center_orig_image_y - np.trunc(size1y/2))
        
    image_gen_crop =  image[idx_x_lower:idx_x_upper, idx_y_lower:idx_y_upper]


    ## Crop small scenes from the first crop
    for i in range(scenesx):
        for j in range(scenesy): 
            crop = image_gen_crop[size_crop[0]*i:size_crop[0]*(i+1), size_crop[1]*j:size_crop[1]*(j+1)]
            scenes_array[:,:,i*scenesy + j] = crop
    
    return scenes_array


aux = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])


for i in range(5):
    print(i)