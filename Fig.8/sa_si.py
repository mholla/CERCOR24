#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:42:55 2023

@author: nagehan
"""

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

si =     np.array([0.208,0.199,0.175,0.169,0.155,0.140,0.129,0.122,0.112,0.104,0.096,0.091,0.086,0.081,0.071])
si_std = np.array([0.005,0.012,0.024,0.003,0.011,0.010,0.010,0.010,0.008,0.010,0.009,0.010,0.009,0.008,0.007])

# Onto vs phylo: gyrification index

sap = np.array([210, 225, 228, 253, 261, 718, 750, 761, 2173]) #surface area phylogenesis
# sap1 = np.array([14, 53, 98])
sap2 = np.array([210, 225, 228, 253, 261])
sap3 = np.array([718, 750, 761])
sap4 = np.array([2173])


sip = np.array([0.18, 0.19, 0.20, 0.19, 0.22, 0.18, 0.16, 0.17, 0.11]) #gyrification index phylogenesis
# sip1 = np.array([1.03, 1.35, 1.53])
sip2 = np.array([0.18, 0.19, 0.20, 0.18, 0.22])
sip3 = np.array([0.18, 0.16, 0.17])
sip4 = np.array([0.10])

sao = np.array([248,279,333,396,405,501,577,616,717,769,881,917,968,1034,1132]) #surface area ontogenesis
gio = np.array([0.208,0.199,0.175,0.169,0.155,0.140,0.129,0.122,0.112,0.104,0.096,0.091,0.086,0.081,0.071]) #gyrification index ontogenesis

slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(np.log(sap), sip)
slope2, intercept2, r_value2, p_value2, std_err2 = scipy.stats.linregress(np.log(sao), si)

plt.plot(sap, slope1 * np.log(sap) + intercept1, '--', color = 'gray')
plt.plot(sao, slope2 * np.log(sao) + intercept2, '--', color = 'gray')

# plt.plot(sap1, gip1, 'o', color='#fde725', markersize=5, label='medium - NHP', alpha = 0.8) 
plt.plot(sap2, sip2, 'o', color='#35b779', markersize=5, label='medium - NHP', alpha = 0.8) 
plt.plot(sap3, sip3, 'o', color='#31688e', markersize=5, label='large - NHP', alpha = 0.8) 
plt.plot(sap4, sip4, 'o', color='#440154', markersize=5, label='xlarge', alpha = 0.8) 

plt.plot(sao, si, 'o', color='#440154', markersize=5, label='human growth', alpha = 0.7) 

plt.legend()
# plt.xscale('log') 
#plt.yscale('log')
plt.savefig('sa_si.jpg', dpi=500, bbox_inches='tight')

