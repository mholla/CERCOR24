#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 14:06:52 2022

@author: nagehan
"""

import scipy
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

gi =     np.array([1.27,1.34,1.49,1.62,1.64,1.82,1.97,2.07,2.24,2.34,2.50,2.55,2.63,2.72,2.81])
gi_std = np.array([0.02,0.05,0.16,0.02,0.09,0.08,0.09,0.09,0.12,0.11,0.09,0.10,0.11,0.12,0.10])


# Onto vs phylo: gyrification index

sap = np.array([210, 225, 228, 253, 261, 718, 750, 761, 2173]) #surface area phylogenesis
# sap1 = np.array([14, 53, 98])
sap2 = np.array([210, 225, 228, 253, 261])
sap3 = np.array([718, 750, 761])
sap4 = np.array([2173])


gip = np.array([1.69, 1.85, 1.81, 1.83, 1.77, 2.26, 2.33, 2.21, 2.53]) #gyrification index phylogenesis
# gip1 = np.array([1.03, 1.35, 1.53])
gip2 = np.array([1.69, 1.85, 1.81, 1.83, 1.77])
gip3 = np.array([2.26, 2.33, 2.21])
gip4 = np.array([2.53])

sao = np.array([248,279,333,396,405,501,577,616,717,769,881,917,968,1034,1132]) #surface area ontogenesis
gio = np.array([1.27,1.34,1.49,1.62,1.64,1.82,1.97,2.07,2.24,2.34,2.50,2.55,2.63,2.72,2.81]) #gyrification index ontogenesis

slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(np.log(sap), gip)
slope2, intercept2, r_value2, p_value2, std_err2 = scipy.stats.linregress(np.log(sao), gio)

plt.plot(sap, slope1 * np.log(sap) + intercept1, '--', color = 'gray')
plt.plot(sao, slope2 * np.log(sao) + intercept2, '--', color = 'gray')

# plt.plot(sap1, gip1, 'o', color='#fde725', markersize=5, label='medium - NHP', alpha = 0.8) 
plt.plot(sap2, gip2, 'o', color='#35b779', markersize=5, label='medium - NHP', alpha = 0.8) 
plt.plot(sap3, gip3, 'o', color='#31688e', markersize=5, label='large - NHP', alpha = 0.8) 
plt.plot(sap4, gip4, 'o', color='#440154', markersize=5, label='xlarge', alpha = 0.8) 

plt.plot(sao, gio, 'o', color='#440154', markersize=5, label='human growth', alpha = 0.7) 

plt.legend()
# plt.xscale('log') 
#plt.yscale('log')
plt.savefig('sa_gi.jpg', dpi=500, bbox_inches='tight')

