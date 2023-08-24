#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:26:47 2023

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

# Read SI
SI1 = '/Users/.../Fig.2/SI_all_gw29.asc'
with open(SI1,'r') as SI1_file:
    SI1_lines = SI1_file.readlines()
si1 = np.zeros(len(SI1_lines))
for i in range(len(SI1_lines)):
    SI1_data = SI1_lines[i].split()
    si1[i] = SI1_data[0]
    
SI2 = '/Users/.../Fig.2/SI_all_gw36.asc'
with open(SI2,'r') as SI2_file:
    SI2_lines = SI2_file.readlines()
si2 = np.zeros(len(SI2_lines))
for i in range(len(SI2_lines)):
    SI2_data = SI2_lines[i].split()
    si2[i] = SI2_data[0]
    
SI3 = '/Users/.../Fig.2/SI_all_gw43.asc'
with open(SI3,'r') as SI3_file:
    SI3_lines = SI3_file.readlines()
si3 = np.zeros(len(SI3_lines))
for i in range(len(SI3_lines)):
    SI3_data = SI3_lines[i].split()
    si3[i] = SI3_data[0]
        
si1 = si1[si1 != 0]
si2 = si2[si2 != 0]
si3 = si3[si3 != 0]

plt.figure()
sns.kdeplot(si1, color='#c6dbef',shade=False, label='29 GW',bw_adjust=3, cut=0)
sns.kdeplot(si2, color='#4292c6',shade=False, label='36 GW',bw_adjust=4, cut=0)
sns.kdeplot(si3, color='#08519c',shade=False, label='43 GW',bw_adjust=4, cut=0)
plt.axvline(si2.mean(), color='#4292c6', ls=':')
plt.axvline(si1.mean(), color='#c6dbef', ls=':')
plt.axvline(si3.mean(), color='#08519c', ls=':')
plt.legend(loc='best',frameon=False)
plt.savefig('si', dpi=500)
    
# Read CT
CT1 = '/Users/.../Fig.2/t_all_gw29.asc'
with open(CT1,'r') as CT1_file:
    CT1_lines = CT1_file.readlines()
ct1 = np.zeros(len(CT1_lines))
for i in range(len(CT1_lines)):
    CT1_data = CT1_lines[i].split()
    ct1[i] = CT1_data[0]
    
CT2 = '/Users/.../Fig.2/t_all_gw36.asc'
with open(CT2,'r') as CT2_file:
    CT2_lines = CT2_file.readlines()
ct2 = np.zeros(len(CT2_lines))
for i in range(len(CT2_lines)):
    CT2_data = CT2_lines[i].split()
    ct2[i] = CT2_data[0]
    
CT3 = '/Users/.../Fig.2/t_all_gw43.asc'
with open(CT3,'r') as CT3_file:
    CT3_lines = CT3_file.readlines()
ct3 = np.zeros(len(CT3_lines))
for i in range(len(CT3_lines)):
    CT3_data = CT3_lines[i].split()
    ct3[i] = CT3_data[0]
    
ct1[ct1 < 0.1] = 0
ct1[ct1 > 5.0] = 0
ct1 = ct1[ct1 != 0]

ct2[ct2 < 0.1] = 0
ct2[ct2 > 5.0] = 0
ct2 = ct2[ct2 != 0]

ct3[ct3 < 0.1] = 0
ct3[ct3 > 5.0] = 0
ct3 = ct3[ct3 != 0]

plt.figure()
sns.kdeplot(ct1, color='#c6dbef',shade=False, label='GW 29',bw_adjust=4, cut=0)
sns.kdeplot(ct2, color='#4292c6',shade=False, label='GW 36',bw_adjust=3, cut=0)
sns.kdeplot(ct3, color='#08519c',shade=False, label='GW 43',bw_adjust=3, cut=0)
plt.axvline(ct1.mean(), color='#c6dbef', ls=':')
plt.axvline(ct2.mean(), color='#4292c6', ls=':')
plt.axvline(ct3.mean(), color='#08519c', ls=':')
plt.legend(loc='upper left',frameon=False)
plt.xlim(0.2, 2)
plt.savefig('ct', dpi=500)

# Read SD
SD1 = '/Users/.../Fig.2/sd_all_gw29.asc'
with open(SD1,'r') as SD1_file:
    SD1_lines = SD1_file.readlines() 
sd1 = np.zeros(len(SD1_lines)) 
for i in range(len(SD1_lines)):
    SD1_data = SD1_lines[i].split()
    sd1[i] = SD1_data[0]
    
SD2 = '/Users/.../Fig.2/sd_all_gw36.asc'
with open(SD2,'r') as SD2_file:
    SD2_lines = SD2_file.readlines()
sd2 = np.zeros(len(SD2_lines))
for i in range(len(SD2_lines)):
    SD2_data = SD2_lines[i].split()
    sd2[i] = SD2_data[0]
    
SD3 = '/Users/.../Fig.2/sd_all_gw43.asc'
with open(SD3,'r') as SD3_file:
    SD3_lines = SD3_file.readlines()
sd3 = np.zeros(len(SD3_lines))
for i in range(len(SD3_lines)):
    SD3_data = SD3_lines[i].split()
    sd3[i] = SD3_data[0]
    
sd1 = sd1[sd1 != 0]
sd2 = sd2[sd2 != 0]
sd3 = sd3[sd3 != 0]

plt.figure()
sns.kdeplot(sd1, color='#c6dbef',shade=False, label='GW 29',bw_adjust=7, cut=0)
sns.kdeplot(sd2, color='#4292c6',shade=False, label='GW 36',bw_adjust=9, cut=0)
sns.kdeplot(sd3, color='#08519c',shade=False, label='GW 43',bw_adjust=10, cut=0)
plt.axvline(sd1.mean(), color='#c6dbef', ls=':')
plt.axvline(sd2.mean(), color='#4292c6', ls=':')
plt.axvline(sd3.mean(), color='#08519c', ls=':')
plt.legend(loc='best',frameon=False)
plt.savefig('sd', dpi=500)