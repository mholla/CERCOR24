#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 20:39:24 2023

@author: nagehan
"""

import scipy
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pds
import seaborn as sns
from scipy.stats import logistic
from scipy.optimize import curve_fit

                  #29    30    31    32    33    34    35    36    37    38    39    40    41    42    43
si =     np.array([0.208,0.199,0.175,0.169,0.155,0.140,0.129,0.122,0.112,0.104,0.096,0.091,0.086,0.081,0.071])
si_std = np.array([0.005,0.012,0.024,0.003,0.011,0.010,0.010,0.010,0.008,0.010,0.009,0.010,0.009,0.008,0.007])

si1 =     np.array([0.208,0.199,0.175])
si_std1 = np.array([0.005,0.012,0.024])

si2 =     np.array([0.169,0.155,0.140,0.129,0.122,0.112])
si_std2 = np.array([0.003,0.011,0.010,0.010,0.010,0.008])

si3 =     np.array([0.104,0.096,0.091,0.086,0.081,0.071])
si_std3 = np.array([00.010,0.009,0.010,0.009,0.008,0.007])

#                  29   30   31   32   33   34   35   36   37   38   39   40   41   42   43
ctr =     np.array([1.04,1.08,1.11,1.13,1.13,1.15,1.17,1.18,1.18,1.20,1.21,1.20,1.21,1.21,1.21])
ctr_std = np.array([0.05,0.02,0.00,0.01,0.02,0.02,0.03,0.02,0.02,0.03,0.03,0.02,0.02,0.02,0.01])

ctr1 =     np.array([1.04,1.08,1.11])
ctr_std1 = np.array([0.05,0.02,0.00])

ctr2 =     np.array([1.13,1.13,1.15,1.17,1.18,1.18])
ctr_std2 = np.array([0.01,0.02,0.02,0.03,0.02,0.02])

ctr3 =     np.array([1.20,1.21,1.20,1.21,1.21,1.21])
ctr_std3 = np.array([0.03,0.03,0.02,0.02,0.02,0.01])

###############################################################################
p = np.polyfit(si, ctr, 2, full=True)

xfit = np.linspace(min(si), max(si), 15)
yfit = np.polyval(p[0], xfit)

plt.scatter(si3, ctr3, color='#000000', marker='o', s=60, alpha=1, label="tertiary")
plt.scatter(si2, ctr2, color='#737373', marker='o', s=60, alpha=1, label="secondary")
plt.scatter(si1, ctr1, color='#d9d9d9', marker='o', s=60, alpha=1, label="primary")
plt.errorbar(si1, ctr1, xerr = si_std1, yerr = ctr_std1, fmt='o',
              color = '#d9d9d9', ecolor = '#d9d9d9', markersize=4, elinewidth = 0.5, capsize=0)
plt.errorbar(si2, ctr2, xerr = si_std2, yerr = ctr_std2, fmt='o',
              color = '#737373', ecolor = '#737373', markersize=4, elinewidth = 0.5, capsize=0)
plt.errorbar(si3, ctr3, xerr = si_std3, yerr = ctr_std3, fmt='o',
              color = '#000000', ecolor = '#000000', markersize=4, elinewidth = 0.5, capsize=0)

plt.plot(xfit, yfit, '--', color = 'gray', zorder=0)
plt.savefig('ctr.jpg', dpi=500, bbox_inches='tight')

###############################################################################
# Calculate r_squared

yhat = np.polyval(p[0], si) 
y = ctr                 
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
r2_ctr = ssreg / sstot

