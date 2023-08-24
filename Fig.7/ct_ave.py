#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:59:42 2023

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

                    #  29    30    31    32    33    34    35    36    37    38    39    40    41    42    43
ct_gyr =     np.array([1.162,1.185,1.201,1.123,1.178,1.164,1.178,1.174,1.187,1.194,1.217,1.226,1.237,1.243,1.288])
ct_gyr_std = np.array([0.049,0.032,0.037,0.017,0.057,0.023,0.060,0.059,0.048,0.037,0.045,0.053,0.050,0.044,0.046])

                    #  29    30    31    32    33    34    35    36    37    38    39    40    41    42    43
ct_gyr1 =     np.array([1.162,1.185,1.201])
ct_gyr_std1 = np.array([0.049,0.032,0.037])

                    #   32    33    34    35    36    37 
ct_gyr2 =     np.array([1.123,1.178,1.164,1.178,1.174,1.187])
ct_gyr_std2 = np.array([0.017,0.057,0.023,0.060,0.059,0.048])

                    #   38    39    40    41    42    43
ct_gyr3 =     np.array([1.194,1.217,1.226,1.237,1.243,1.288])
ct_gyr_std3 = np.array([0.037,0.045,0.053,0.050,0.044,0.046])

                    #  29    30    31    32    33    34    35    36    37    38    39    40    41    42    43
ct_sad =     np.array([1.148,1.183,1.183,1.093,1.143,1.117,1.120,1.112,1.118,1.114,1.126,1.135,1.142,1.147,1.183])
ct_sad_std = np.array([0.042,0.031,0.046,0.013,0.054,0.018,0.053,0.054,0.049,0.036,0.041,0.049,0.046,0.042,0.044])

ct_sad1 =     np.array([1.148,1.183,1.183])
ct_sad_std1 = np.array([0.042,0.031,0.046])

ct_sad2 =     np.array([1.093,1.143,1.117,1.120,1.112,1.118])
ct_sad_std2 = np.array([0.013,0.054,0.018,0.053,0.054,0.049])

ct_sad3 =     np.array([1.114,1.126,1.135,1.142,1.147,1.183])
ct_sad_std3 = np.array([0.036,0.041,0.049,0.046,0.042,0.044])

                     #  29    30    31    32    33    34    35    36    37    38    39    40    41    42    43
ct_sulc =    np.array([1.105,1.102,1.085,0.994,1.039,1.011,1.004,0.999,1.004,0.999,1.009,1.019,1.026,1.031,1.063])
ct_sulc_std =np.array([0.010,0.032,0.034,0.010,0.050,0.017,0.048,0.049,0.044,0.038,0.041,0.047,0.044,0.041,0.039])

ct_sulc1 =    np.array([1.105,1.102,1.085])
ct_sulc_std1 =np.array([0.010,0.032,0.034])

ct_sulc2 =    np.array([0.994,1.039,1.011,1.004,0.999,1.004])
ct_sulc_std2 =np.array([0.010,0.050,0.017,0.048,0.049,0.044])

ct_sulc3 =    np.array([0.999,1.009,1.019,1.026,1.031,1.063])
ct_sulc_std3 =np.array([0.038,0.041,0.047,0.044,0.041,0.039])

###############################################################################
p = np.polyfit(si, ct_sulc, 2, full=True)

xfit = np.linspace(min(si), max(si), 15)
yfit = np.polyval(p[0], xfit)

plt.figure()
plt.scatter(si3, ct_sulc3, color='#000000', marker='o', s=60, alpha=1, label="tertiary")
plt.scatter(si2, ct_sulc2, color='#737373', marker='o', s=60, alpha=1, label="secondary")
plt.scatter(si1, ct_sulc1, color='#d9d9d9', marker='o', s=60, alpha=1, label="primary")
plt.errorbar(si1, ct_sulc1, xerr = si_std1, yerr = ct_sulc_std1, fmt='o',
              color = '#d9d9d9', ecolor = '#d9d9d9', markersize=4, elinewidth = 0.5, capsize=0)
plt.errorbar(si2, ct_sulc2, xerr = si_std2, yerr = ct_sulc_std2, fmt='o',
              color = '#737373', ecolor = '#737373', markersize=4, elinewidth = 0.5, capsize=0)
plt.errorbar(si3, ct_sulc3, xerr = si_std3, yerr = ct_sulc_std3, fmt='o',
              color = '#000000', ecolor = '#000000', markersize=4, elinewidth = 0.5, capsize=0)

plt.plot(xfit, yfit, '--', color = 'gray',zorder=0)

###############################################################################
# Calculate r_squared

yhat = np.polyval(p[0], si)     
y = ct_sulc                     
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
r2_sulc = ssreg / sstot

###############################################################################
p = np.polyfit(si, ct_gyr, 2, full=True)

xfit = np.linspace(min(si), max(si), 15)
yfit = np.polyval(p[0], xfit)

plt.scatter(si3, ct_gyr3, color='#000000', marker='s', s=60, alpha=1, label="tertiary")
plt.scatter(si2, ct_gyr2, color='#737373', marker='s', s=60, alpha=1, label="secondary")
plt.scatter(si1, ct_gyr1, color='#d9d9d9', marker='s', s=60, alpha=1, label="primary")
plt.errorbar(si1, ct_gyr1, xerr = si_std1, yerr = ct_gyr_std1, fmt='o',
              color = '#d9d9d9', ecolor = '#d9d9d9', markersize=4, elinewidth = 0.5, capsize=0)
plt.errorbar(si2, ct_gyr2, xerr = si_std2, yerr = ct_gyr_std2, fmt='o',
              color = '#737373', ecolor = '#737373', markersize=4, elinewidth = 0.5, capsize=0)
plt.errorbar(si3, ct_gyr3, xerr = si_std3, yerr = ct_gyr_std3, fmt='o',
              color = '#000000', ecolor = '#000000', markersize=4, elinewidth = 0.5, capsize=0)

plt.plot(xfit, yfit, '--', color = 'gray',zorder=0)

###############################################################################
# Calculate r_squared

yhat = np.polyval(p[0], si)   
y = ct_gyr                   
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
r2_gyr = ssreg / sstot

###############################################################################
p = np.polyfit(si, ct_sad, 2, full=True)

xfit = np.linspace(min(si), max(si), 15)
yfit = np.polyval(p[0], si)

plt.scatter(si3, ct_sad3, color='#000000', marker='D', s=60, alpha=1, label="tertiary")
plt.scatter(si2, ct_sad2, color='#737373', marker='D', s=60, alpha=1, label="secondary")
plt.scatter(si1, ct_sad1, color='#d9d9d9', marker='D', s=60, alpha=1, label="primary")

plt.errorbar(si1, ct_sad1, xerr = si_std1, yerr = ct_sad_std1, fmt='o',
              color = '#d9d9d9', ecolor = '#d9d9d9', markersize=4, elinewidth = 0.5, capsize=0)
plt.errorbar(si2, ct_sad2, xerr = si_std2, yerr = ct_sad_std2, fmt='o',
              color = '#737373', ecolor = '#737373', markersize=4, elinewidth = 0.5, capsize=0)
plt.errorbar(si3, ct_sad3, xerr = si_std3, yerr = ct_sad_std3, fmt='o',
              color = '#000000', ecolor = '#000000', markersize=4, elinewidth = 0.5, capsize=0)

plt.plot(xfit, yfit, '--', color = 'gray',zorder=0)
plt.savefig('ct_ave.jpg', dpi=500, bbox_inches='tight')

###############################################################################
# Calculate r_squared

yhat = np.polyval(p[0], si)  
y = ct_sad                 
ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
r2_sad = ssreg / sstot



