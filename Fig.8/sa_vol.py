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

#Human surface area and volume from gestation week 29 to 43 and ages 7-13, 14-22, and 23-56

#                      29  30  31  32  33  34  35  36  37  38  39  40  41  42   43   7-13 14-22 23-56 (ages, GW, years)
sa =         np.array([248,279,333,396,405,501,577,616,717,769,881,917,968,1034,1132,2163,2182,2173])

#                      29  30  31  32  33  34  35  36  37  38  39  40  41  42   43 7-13 14-22 23-56 (ages, GW, years)
vol =        np.array([125,139,148,174,180,199,218,223,247,256,281,287,298,310,337,1047,1063,1068])

# Onto vs phylo: volume

sap = np.array([210, 225, 228, 253, 261, 718, 750, 761, 2173]) #surface area phylogenesis
#sap1 = np.array([14, 53, 98])
sap2 = np.array([210, 225, 228, 253, 261])
sap3 = np.array([718, 750, 761])
sap4 = np.array([2173])


volp = np.array([59, 84, 66, 68, 87, 280, 229, 224, 1053]) # volume phylogenesis
#volp1 = np.array([3, 11, 23])
volp2 = np.array([59, 84, 66, 68, 87])
volp3 = np.array([280, 229, 224])
volp4 = np.array([1053])

sao = np.array([248,279,333,396,405,501,577,616,717,769,881,917,968,1034,1132]) #surface area ontogenesis
volo = np.array([125,139,148,174,180,199,218,223,247,256,281,287,298,310,337]) # volume otogenesis

slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(np.log(sap), np.log(volp))
slope2, intercept2, r_value2, p_value2, std_err2 = scipy.stats.linregress(np.log(sao), np.log(volo))

plt.plot(sap, np.exp(slope1 * np.log(sap) + intercept1), '--', color = 'gray', alpha=0.8)
plt.plot(sao, np.exp(slope2 * np.log(sao) + intercept2), '--', color = 'gray', alpha=0.8)

# plt.plot(sap, np.exp(1.5*np.log(sap) + 1.65*intercept1), '--', color = 'gray', alpha=0.3)
plt.plot(sao, np.exp(1.5*np.log(sao) + -2.46*intercept2), '--', color = 'gray', alpha=0.3)

#plt.plot(sap1, volp1, 'o', color='#fde725', markersize=5, label='small') 
plt.plot(sap2, volp2, 'o', color='#35b779', markersize=5, label='medium - NHP', alpha = 0.8) 
plt.plot(sap3, volp3, 'o', color='#31688e', markersize=5, label='large - NHP', alpha = 0.8) 
plt.plot(sap4, volp4, 'o', color='#440154', markersize=5, label='large - NHP', alpha = 0.8) 

plt.plot(sao, volo, 'o', color='#440154', markersize=5, label='human growth', alpha = 0.7) 

plt.legend()
plt.xscale('log') 
plt.yscale('log') 
plt.savefig('sa_vol.jpg', dpi=500, bbox_inches='tight')

