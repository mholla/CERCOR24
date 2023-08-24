#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:38:50 2023

@author: nagehan
"""

import scipy
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

#                  29   30   31   32   33   34   35   36   37   38   39   40   41   42   43
ctr =     np.array([1.04,1.08,1.11,1.13,1.13,1.15,1.17,1.18,1.18,1.20,1.21,1.20,1.21,1.21,1.21])
ctr_std = np.array([0.05,0.02,0.00,0.01,0.02,0.02,0.03,0.02,0.02,0.03,0.03,0.02,0.02,0.02,0.01])

# Onto vs phylo: cortical thickness

sap = np.array([210, 225, 228, 253, 261, 718, 750, 761, 2173]) #surface area phylogenesis
# sap1 = np.array([14, 53, 98])
sap2 = np.array([210, 225, 228, 253, 261])
sap3 = np.array([718, 750, 761])
sap4 = np.array([2173])


ctrp = np.array([1.23, 1.33, 1.18, 1.19, 1.19, 1.19, 1.35, 1.39, 1.3]) #cortical thickness ratio phylogenesis
# ctrp1 = np.array([1.05, 1.19, 1.18])
ctrp2 = np.array([1.23, 1.33, 1.18, 1.19, 1.19])
ctrp3 = np.array([1.19, 1.35, 1.39])
ctrp4 = np.array([1.3])

sao = np.array([248,279,333,396,405,501,577,616,717,769,881,917,968,1034,1132]) #surface area ontogenesis
ctro = np.array([1.04,1.08,1.11,1.13,1.13,1.15,1.17,1.18,1.18,1.20,1.21,1.20,1.21,1.21,1.21])

slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(np.log(sap), np.log(ctrp))
slope2, intercept2, r_value2, p_value2, std_err2 = scipy.stats.linregress(np.log(sao), np.log(ctro))

plt.plot(sap, np.exp(slope1 * np.log(sap) + intercept1), '--', color = 'gray')
plt.plot(sao, np.exp(slope2 * np.log(sao) + intercept2), '--', color = 'gray')
# plt.plot(sap, slope1 * np.log(sap) + 192*intercept1, '--', color = 'gray', alpha=0.3)
#plt.plot(sao, np.exp(0.5*np.log(sao) + 0.76*intercept2), '--', color = 'gray', alpha=0.3)

# plt.plot(sap1, ctp1, 'o', color='#fde725', markersize=5, label='small', alpha = 0.8) 
plt.plot(sap2, ctrp2, 'o', color='#35b779', markersize=5, label='medium', alpha = 0.8) 
plt.plot(sap3, ctrp3, 'o', color='#31688e', markersize=5, label='large', alpha = 0.8) 
plt.plot(sap4, ctrp4, 'o', color='#440154', markersize=5, label='large', alpha = 0.8) 

plt.plot(sao, ctro, 'o', color='#440154', markersize=5, label='human growth', alpha = 0.7) 

#plt.legend()
# plt.xscale('log') 
# plt.yscale('log')
# plt.ylim(0,3)
plt.savefig('sa_ctr.jpg', dpi=500, bbox_inches='tight')