#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:09:16 2023

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

sd = [] 
SD = '/Users/.../sd.txt'
with open(SD,'r') as SD_file:
    SD_lines = SD_file.readlines() 
sd = np.zeros(len(SD_lines))  
for i in range(len(SD_lines)):
    SD_data = SD_lines[i].split()
    sd[i] = SD_data[0]
    
pri_sd = np.array(sd[0:8]) # between gw29-31  
sec_sd = np.array(sd[8:99]) # between gw32-37  
ter_sd = np.array(sd[99:]) # from gw38 and forward  

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

###############################################################################

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(si, sd)

p = np.polyfit(si, np.log(sd), 1)

a = np.exp(p[1])
b = p[0]
x_fitted = np.linspace(np.min(si), np.max(si), 100)
y_fitted = a * np.exp(b * x_fitted)

plt.figure()
plt.plot(ter_si, ter_sd, 'o', color='#000000', ms=4, mec="red", alpha=0.7, markeredgewidth=0)
plt.plot(sec_si, sec_sd, 'o', color='#737373', ms=4, mec="red", alpha=0.7, markeredgewidth=0)
plt.plot(pri_si, pri_sd, 'o', color='#d9d9d9', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(x_fitted, y_fitted, '--', color = 'gray', alpha=1)

plt.savefig('si_sd.jpg', dpi=500, bbox_inches='tight')

###############################################################################

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(si, sd)

p = np.polyfit(si, sd, 1)

a = p[1]
b = p[0]
x_fitted = np.linspace(np.min(si), np.max(si), 100)
y_fitted = a + b * x_fitted

plt.figure()
plt.plot(x_fitted, y_fitted, '--', color = 'gray', alpha=1)
plt.scatter(ter_si, ter_sd, color='#000000', marker='o', s=10, alpha=1, label="tertiary")
plt.scatter(sec_si, sec_sd, color='#737373', marker='o', s=10, alpha=1, label="secondary")
plt.scatter(pri_si, pri_sd, color='#d9d9d9', marker='o', s=10, alpha=1, label="primary")

plt.savefig('si_sd.jpg', dpi=500, bbox_inches='tight')

###################################################
# Calculate r_squared
yhat = a + b * si
yhat = a * np.exp(b * si)
y = sd                  
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
r2_sd = ssreg / sstot