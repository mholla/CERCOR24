#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:56:56 2023

@author: nagehan
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import scipy.stats as stats
import seaborn as sns
import scipy
import scipy.stats
import pandas as pds
import seaborn as sns
from scipy.stats import logistic
from scipy.optimize import curve_fit

#################################################################################
si = [] 
SI = '/Users/.../si_pial.txt'
with open(SI,'r') as SI_file:
    SI_lines = SI_file.readlines() 
si = np.zeros(len(SI_lines))  
for i in range(len(SI_lines)):
    SI_data = SI_lines[i].split()
    si[i] = SI_data[0]

gi = [] 
GI = '/Users/.../gi.txt'
with open(GI,'r') as GI_file:
    GI_lines = GI_file.readlines() 
gi = np.zeros(len(GI_lines))  
for i in range(len(GI_lines)):
    GI_data = GI_lines[i].split()
    gi[i] = GI_data[0]
    
###############################################################################
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(si, gi)

p = np.polyfit(si, gi, 1)

a = p[1]
b = p[0]
x_fitted = np.linspace(np.min(si), np.max(si), 501)
y_fitted = a + b * x_fitted

plt.figure()
plt.plot(si, gi, 'o',color="black", ms=4, mec="red", alpha=0.8, markeredgewidth=0)
plt.plot(x_fitted, y_fitted, '--', color = 'gray', alpha=1, linewidth=2)
plt.savefig('si_gi.jpg', dpi=500, bbox_inches='tight')

###################################################
# Calculate r_squared
yhat = a + b * si  
y = gi                  
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
r2_gi = ssreg / sstot




