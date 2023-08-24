#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 23:07:20 2023

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


si = [] 
SI = '/Users/.../si_pial.txt'
with open(SI,'r') as SI_file:
    SI_lines = SI_file.readlines() 
si = np.zeros(len(SI_lines))  
for i in range(len(SI_lines)):
    SI_data = SI_lines[i].split()
    si[i] = SI_data[0]
    
pri_si = np.array(si[0:8]) # between gw29-31  
sec_si = np.array(si[8:99]) # between gw32-37  
ter_si = np.array(si[99:]) # from gw38 and forward  

siw = [] 
SIw = '/Users/.../si_white.txt'
with open(SIw,'r') as SIw_file:
    SIw_lines = SIw_file.readlines() 
siw = np.zeros(len(SIw_lines))  
for i in range(len(SIw_lines)):
    SIw_data = SIw_lines[i].split()
    siw[i] = SIw_data[0]
    
pri_siw = np.array(siw[0:8]) # between gw29-31  
sec_siw = np.array(siw[8:99]) # between gw32-37  
ter_siw = np.array(siw[99:]) # from gw38 and forward  

vol = [] 
VOL = '/Users/.../vol.txt'
with open(VOL,'r') as VOL_file:
    VOL_lines = VOL_file.readlines() 
vol = np.zeros(len(VOL_lines))  
for i in range(len(VOL_lines)):
    VOL_data = VOL_lines[i].split()
    vol[i] = VOL_data[0]
    
pri_vol = np.array(vol[0:8]) # between gw29-31  
sec_vol = np.array(vol[8:99]) # between gw32-37  
ter_vol = np.array(vol[99:]) # from gw38 and forward  

volw = [] 
VOLW = '/Users/.../vol_white.txt'
with open(VOLW,'r') as VOLW_file:
    VOLW_lines = VOLW_file.readlines() 
volw = np.zeros(len(VOLW_lines))  
for i in range(len(VOLW_lines)):
    VOLW_data = VOLW_lines[i].split()
    volw[i] = VOLW_data[0]
    
pri_volw = np.array(volw[0:8]) # between gw29-31  
sec_volw = np.array(volw[8:99]) # between gw32-37  
ter_volw = np.array(volw[99:]) # from gw38 and forward  

###############################################################################

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(si, np.log(vol))

p = np.polyfit(si, np.log(vol), 1)

a = np.exp(p[1])
b = p[0]
x_fitted = np.linspace(np.min(si), np.max(si), 100)
y_fitted = a * np.exp(b * x_fitted)

plt.figure()
plt.plot(ter_si, ter_vol, 'o', color='#000000', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(sec_si, sec_vol, 'o', color='#737373', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(pri_si, pri_vol, 'o', color='#d9d9d9', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(x_fitted, y_fitted, '--', color = 'gray', alpha=1)

###############################################################################
# Calculate r_squared
yhat = a * np.exp(b * si)
y = vol                  
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
r2_vol = ssreg / sstot

###############################################################################

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(siw, np.log(volw))

p = np.polyfit(siw, np.log(volw), 1)

a = np.exp(p[1])
b = p[0]
x_fitted = np.linspace(np.min(siw), np.max(siw), 100)
y_fitted = a * np.exp(b * x_fitted)

plt.plot(ter_siw, ter_volw, 'o', color='#000000', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(sec_siw, sec_volw, 'o', color='#737373', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(pri_siw, pri_volw, 'o', color='#d9d9d9', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(x_fitted, y_fitted, '--', color = 'gray', alpha=1)

plt.savefig('si_vol.jpg', dpi=500, bbox_inches='tight')

###############################################################################
# Calculate r_squared
y_fitted = a * np.exp(b * siw)
yhat = y_fitted    
y = volw                  
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
r2_volw = ssreg / sstot
