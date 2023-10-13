#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 12:03:48 2022

@author: nagehan
"""

import scipy
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


sd =     np.array([1.70,1.85,2.50,2.61,2.83,3.32,3.69,3.86,4.32,4.47,4.87,4.99,5.20,5.43,5.76])
sd_std = np.array([0.11,0.18,0.56,0.06,0.24,0.24,0.22,0.31,0.24,0.28,0.22,0.24,0.24,0.25,0.25])

# Onto vs phylo: sulcal depth

sap = np.array([210, 225, 228, 253, 261, 718, 750, 761, 2173]) #surface area phylogenesis
# sap1 = np.array([14, 53, 98])
sap2 = np.array([210, 225, 228, 253, 261])
sap3 = np.array([718, 750, 761])
sap4 = np.array([2163, 2182, 2173])
sap4 = np.array([2163, 2182, 2173])
# sap4 = np.array([2173])

sdp = np.array([2.40, 2.32, 2.41, 2.77, 2.55, 4.17, 4.04, 3.83, 6.92]) #sulcal depth phylogenesis
# sdp1 = np.array([0.59, 1.44, 1.74])
sdp2 = np.array([2.40, 2.32, 2.41, 2.77, 2.55])
sdp3 = np.array([4.17, 4.04, 3.83])
sdp4 = np.array([6.91, 6.82, 6.64])
# sdp4 = np.array([6.92])

sao = np.array([248,279,333,396,405,501,577,616,717,769,881,917,968,1034,1132]) #surface area ontogenesis
sdo = np.array([1.70,1.85,2.50,2.61,2.83,3.32,3.69,3.86,4.32,4.47,4.87,4.99,5.20,5.43,5.76]) #sulcal depth ontogenesis

slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(np.log(sap), np.log(sdp))
slope2, intercept2, r_value2, p_value2, std_err2 = scipy.stats.linregress(np.log(sao), np.log(sdo))

plt.plot(sap, np.exp(slope1 * np.log(sap) + intercept1), '--', color = 'gray')
plt.plot(sao, np.exp(slope2 * np.log(sao) + intercept2), '--', color = 'gray')
plt.plot(sap, np.exp(0.5*np.log(sap) + 1.19*intercept1), '--', color = 'gray', alpha=0.3)
#plt.plot(sao, np.exp(0.5*np.log(sao) + 0.76*intercept2), '--', color = 'gray', alpha=0.3)

# plt.plot(sap1, sdp1, 'o', color='#fde725', markersize=5, label='medium - NHP', alpha = 0.8) 
plt.plot(sap2, sdp2, 'o', color='#35b779', markersize=5, label='medium - NHP', alpha = 0.8) 
plt.plot(sap3, sdp3, 'o', color='#31688e', markersize=5, label='large - NHP', alpha = 0.8) 
plt.plot(sap4, sdp4, 'o', color='#440154', markersize=5, label='xlarge', alpha = 0.8) 

plt.plot(sao, sdo, 'o', color='#440154', markersize=5, label='human growth', alpha = 0.7) 

plt.legend()
plt.xscale('log') 
plt.yscale('log')
plt.savefig('sa_sd.jpg', dpi=500, bbox_inches='tight')
