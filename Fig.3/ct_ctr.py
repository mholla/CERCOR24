#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:22:49 2022

# Cortical thickness

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

#################################################################################
ct = [] 
CT = '/Users/.../ct.txt'
with open(CT,'r') as CT_file:
    CT_lines = CT_file.readlines() 
ct = np.zeros(len(CT_lines))  
for i in range(len(CT_lines)):
    CT_data = CT_lines[i].split()
    ct[i] = CT_data[0]
    
ctr = [] 
CTR = '/Users/.../ctr.txt'
with open(CTR,'r') as CTR_file:
    CTR_lines = CTR_file.readlines() 
ctr = np.zeros(len(CTR_lines))  
for i in range(len(CTR_lines)):
    CTR_data = CTR_lines[i].split()
    ctr[i] = CTR_data[0]
    
age = [] 
AGE = '/Users/.../scan_ages.txt'
with open(AGE,'r') as AGE_file:
    AGE_lines = AGE_file.readlines() 
age = np.zeros(len(AGE_lines))  
for i in range(len(AGE_lines)):
    AGE_data = AGE_lines[i].split()
    age[i] = AGE_data[0]

###############################################################################
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(age, ct) 

m,b = np.polyfit(age, ct, 1)
xfit = np.linspace(min(age), max(age), 501)
yfit = m*xfit + b

# confidence interval
x_mean = np.mean(age)
y_mean = np.mean(ct)
n = age.size                        # number of samples
m = 2                             # number of parameters
dof = n - m                       # degrees of freedom
t = stats.t.ppf(0.975, dof)       # Students statistic of interval confidence
residual = ct - yfit
std_error = (np.sum(residual**2) / dof)**.5
ci = t * std_error * (1/n + (xfit - x_mean)**2 / np.sum((age - x_mean)**2))**.5

plt.figure()
plt.plot(age, ct, 'o',color="black", ms=4, mec="red", alpha=0.7, markeredgewidth=0)
plt.plot(xfit, yfit, '-', linewidth=2, color = 'black')
plt.fill_between(xfit, yfit + ci, yfit - ci, color = 'lightgray', alpha=1, label = '95% confidence interval')
plt.savefig('ct_age.jpg', dpi=500, bbox_inches='tight')

###############################################################################
p = np.polyfit(age, ctr, 2, full=True)

xfit = np.linspace(min(age), max(age), 501)
yfit = np.polyval(p[0], xfit)

# confidence interval
x_mean = np.mean(age)
y_mean = np.mean(ctr)
n = age.size                        # number of samples
m = 2                             # number of parameters
dof = n - m                       # degrees of freedom
t = stats.t.ppf(0.975, dof)       # Students statistic of interval confidence
residual = ctr - yfit
std_error = (np.sum(residual**2) / dof)**.5
ci = t * std_error * (1/n + (xfit - x_mean)**2 / np.sum((age - x_mean)**2))**.5

plt.figure()
plt.plot(age, ctr, 'o',color="black", ms=4, mec="red", alpha=0.7, markeredgewidth=0)
plt.plot(xfit, yfit, '-', linewidth=2, color = 'black')
plt.fill_between(xfit, yfit + ci, yfit - ci, color = 'lightgray', alpha=1, label = '95% confidence interval')
plt.savefig('ctr_age.jpg', dpi=500, bbox_inches='tight')

###################################################
# Calculate r_squared
yhat = np.polyval(p[0], age)     
y = ctr                  
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
r2_ctr = ssreg / sstot

