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
ct_gyr = [] 
CT_gyr = '/Users/.../ct_gyr_pial.txt'
with open(CT_gyr,'r') as CT_gyr_file:
    CT_gyr_lines = CT_gyr_file.readlines() 
ct_gyr = np.zeros(len(CT_gyr_lines))  
for i in range(len(CT_gyr_lines)):
    CT_gyr_data = CT_gyr_lines[i].split()
    ct_gyr[i] = CT_gyr_data[0]
    
ct_sulc = [] 
CT = '/Users/.../ct_sulc_pial.txt'
with open(CT,'r') as CT_file:
    CT_lines = CT_file.readlines() 
ct_sulc = np.zeros(len(CT_lines))  
for i in range(len(CT_lines)):
    CT_data = CT_lines[i].split()
    ct_sulc[i] = CT_data[0]
    
ct_sad = [] 
CT = '/Users/.../ct_sad_pial.txt'
with open(CT,'r') as CT_file:
    CT_lines = CT_file.readlines() 
ct_sad = np.zeros(len(CT_lines))  
for i in range(len(CT_lines)):
    CT_data = CT_lines[i].split()
    ct_sad[i] = CT_data[0]
    
age = [] 
AGE = '/Users/.../scan_ages.txt'
with open(AGE,'r') as AGE_file:
    AGE_lines = AGE_file.readlines() 
age = np.zeros(len(AGE_lines))  
for i in range(len(AGE_lines)):
    AGE_data = AGE_lines[i].split()
    age[i] = AGE_data[0]

###############################################################################
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(age, ct_gyr) 

plt.figure()
#plt.plot(x, m*x + b, '--', color = 'gray')
plt.plot(age, ct_gyr, color='#54278f', marker='o', linestyle='', markersize=3, alpha=0.8)
sns.regplot(age, ct_gyr, color ='#54278f', ci=95, scatter=False, fit_reg=True, scatter_kws={'s':10,'edgecolor':"#54278f"})

#plt.plot(x, m*x + b, '--', color = 'gray')
plt.plot(age, ct_sad, color='#1F9E89', marker='o', linestyle='', markersize=3, alpha=0.8)
sns.regplot(age, ct_sad, color ='#1F9E89', ci=95, scatter=False, fit_reg=True, scatter_kws={'s':10,'edgecolor':"#1F9E89"})

#plt.plot(x, m*x + b, '--', color = 'gray')
plt.plot(age, ct_sulc, color='#FDE725', marker='o', linestyle='', markersize=3, alpha=0.8)
sns.regplot(age, ct_sulc, color ='#FDE725', ci=95, scatter=False, fit_reg=True, scatter_kws={'s':10,'edgecolor':"#FDE725"})

plt.savefig('ct_age.jpg', dpi=500, bbox_inches='tight')
