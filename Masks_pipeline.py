# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:18:14 2022

@author: Pau
"""

import xarray as xr
import os,glob,sys
import numpy as np
import matplotlib.pyplot as plt
#from scipy import interpolate
#from amsrrtm2 import amsr
#from cmod5n import cmod5n_forward
from Mask_script import main  


dataset_path = "../Datasets/"
file_list = []
i = 0
for file in (glob.glob(dataset_path + "*.nc")):
    print("File #{}; Name: {}".format(i, file))
    if (i < 11) and (i > 2):
        ds = xr.open_dataset(file)
        file_list.append(ds);
        main(ds, i)
    i = i+1
    

