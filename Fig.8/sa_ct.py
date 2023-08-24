#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 14:48:22 2022

@author: nagehan
"""

import scipy
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

#                  29   30   31   32   33   34   35   36   37   38   39   40   41   42   43
ct =     np.array([1.15,1.18,1.18,1.09,1.14,1.12,1.12,1.12,1.12,1.12,1.14,1.14,1.15,1.16,1.19])
ct_std = np.array([0.03,0.03,0.04,0.01,0.05,0.02,0.05,0.05,0.05,0.03,0.04,0.05,0.05,0.04,0.04])

# Onto vs phylo: cortical thickness

sap = np.array([210, 225, 228, 253, 261, 718, 750, 761, 2173]) #surface area phylogenesis
# sap1 = np.array([14, 53, 98])
sap2 = np.array([210, 225, 228, 253, 261])
sap3 = np.array([718, 750, 761])
sap4 = np.array([2173])


ctp = np.array([1.53, 1.84, 1.37, 1.54, 1.89, 1.97, 2.31, 2.61, 2.72]) #cortical thickness phylogenesis
# ctp1 = np.array([0.91, 1.37, 1.47])
ctp2 = np.array([1.53, 1.84, 1.37, 1.54, 1.89])
ctp3 = np.array([1.97, 2.31, 2.61])
ctp4 = np.array([2.72])

sao = np.array([248,279,333,396,405,501,577,616,717,769,881,917,968,1034,1132]) #surface area ontogenesis
cto = np.array([1.15,1.18,1.18,1.09,1.14,1.12,1.12,1.12,1.12,1.12,1.14,1.14,1.15,1.16,1.19])

slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(np.log(sap), np.log(ctp))
slope2, intercept2, r_value2, p_value2, std_err2 = scipy.stats.linregress(np.log(sao), np.log(cto))

plt.plot(sap, np.exp(slope1 * np.log(sap) + intercept1), '--', color = 'gray')
plt.plot(sao, np.exp(slope2 * np.log(sao) + intercept2), '--', color = 'gray')
plt.plot(sap, np.exp(0.5*np.log(sap) + 2.15*intercept1), '--', color = 'gray', alpha=0.3)
#plt.plot(sao, np.exp(0.5*np.log(sao) + 0.76*intercept2), '--', color = 'gray', alpha=0.3)

# plt.plot(sap1, ctp1, 'o', color='#fde725', markersize=5, label='small', alpha = 0.8) 
plt.plot(sap2, ctp2, 'o', color='#35b779', markersize=5, label='medium', alpha = 0.8) 
plt.plot(sap3, ctp3, 'o', color='#31688e', markersize=5, label='large', alpha = 0.8) 
plt.plot(sap4, ctp4, 's', color='#440154', markersize=5, label='large', alpha = 0.8) 

plt.plot(sao, cto, 'o', color='#440154', markersize=5, label='human growth', alpha = 0.7) 

#plt.legend()
plt.xscale('log') 
plt.yscale('log')
plt.ylim(0,3)
plt.savefig('sa_ct.jpg', dpi=500, bbox_inches='tight')

