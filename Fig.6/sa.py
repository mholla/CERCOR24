#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:42:43 2023

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

sa = [] 
SA = '/Users/.../sa.txt'
with open(SA,'r') as SA_file:
    SA_lines = SA_file.readlines() 
sa = np.zeros(len(SA_lines))  
for i in range(len(SA_lines)):
    SA_data = SA_lines[i].split()
    sa[i] = SA_data[0]
    
pri_sa = np.array(sa[0:8]) # between gw29-31  
sec_sa = np.array(sa[8:99]) # between gw32-37  
ter_sa = np.array(sa[99:]) # from gw38 and forward

saw = [] 
SAw = '/Users/.../sa_white.txt'
with open(SAw,'r') as SAw_file:
    SAw_lines = SAw_file.readlines() 
saw = np.zeros(len(SAw_lines))  
for i in range(len(SAw_lines)):
    SAw_data = SAw_lines[i].split()
    saw[i] = SAw_data[0]
    
pri_saw = np.array(saw[0:8]) # between gw29-31  
sec_saw = np.array(saw[8:99]) # between gw32-37  
ter_saw = np.array(saw[99:]) # from gw38 and forward

###############################################################################

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(si, np.log(sa))

p = np.polyfit(si, np.log(sa), 1)

a = np.exp(p[1])
b = p[0]
x_fitted = np.linspace(np.min(si), np.max(si), 100)
y_fitted = a * np.exp(b * x_fitted)

plt.figure()
plt.plot(ter_si, ter_sa, 'o', color='#000000', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(sec_si, sec_sa, 'o', color='#737373', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(pri_si, pri_sa, 'o', color='#d9d9d9', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(x_fitted, y_fitted, '--', color = 'gray', alpha=1)

###################################################
# Calculate r_squared
yhat = a * np.exp(b * si)
y = sa                  
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
r2_sa = ssreg / sstot

###############################################################################

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(siw, np.log(saw))

p = np.polyfit(siw, np.log(saw), 1)

a = np.exp(p[1])
b = p[0]
x_fitted = np.linspace(np.min(siw), np.max(siw), 100)
y_fitted = a * np.exp(b * x_fitted)

plt.plot(ter_siw, ter_saw, 'o', color='#000000', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(sec_siw, sec_saw, 'o', color='#737373', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(pri_siw, pri_saw, 'o', color='#d9d9d9', ms=4, mec="red", alpha=1, markeredgewidth=0)
plt.plot(x_fitted, y_fitted, '--', color = 'gray', alpha=1)

plt.savefig('si_sa.jpg', dpi=500, bbox_inches='tight')

###################################################
# Calculate r_squared
yhat = a * np.exp(b * siw)
y = saw                
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
r2_saw = ssreg / sstot