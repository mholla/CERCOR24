#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 18:25:55 2022

@author: nagehan
"""

import matplotlib.pyplot as plt
import numpy as np

# Cortical thickness with respect to shape (convex, concave, saddle)

# Thickness GW29

t1 = [[1.162],
        [1.148],
        [1.105]]

error1 = [[0.010],
          [0.049],
          [0.042]]

# Thickness GW31

t2 = [[1.201],
        [1.183],
        [1.085]]

error2 = [[0.037],
          [0.046],
          [0.034]]

# Thickness GW33

t3 = [[1.178],
        [1.143],
        [1.039]]

error3 = [[0.057],
          [0.054],
          [0.050]]

# Thickness GW35

t4 = [[1.178],
        [1.120],
        [1.004]]

error4 = [[0.060],
          [0.053],
          [0.048]]

# Thickness GW37

t5 = [[1.187],
        [1.118],
        [1.004]]

error5 = [[0.048],
          [0.049],
          [0.044]]

# Thickness GW39

t6 = [[1.217],
        [1.126],
        [1.009]]

error6 = [[0.045],
          [0.041],
          [0.041]]

# Thickness GW41

t7 = [[1.237],
        [1.142],
        [1.026]]

error7 = [[0.050],
          [0.046],
          [0.044]]

# Thickness GW43

t8 = [[1.288],
        [1.183],
        [1.063]]

error8 = [[0.046],
          [0.044],
          [0.039]]

# Thickness Adult

# t9 = [[2.96],
#         [2.70],
#         [2.33]]

# error9 = [[0.17],
#           [0.17],
#           [0.17]]

X = np.array([0])
Y = np.array([1])
Z = np.array([2])
P = np.array([3])
Q = np.array([4])
R = np.array([5])
A = np.array([6])
B = np.array([7])
# C = np.array([8])

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.bar(X - 0.25, t1[2], color = '#FDE725', width = 0.25, yerr = error1[0], bottom=0)
ax.bar(X + 0.00, t1[1], color = '#1F9E89', width = 0.25, yerr = error1[1], bottom=0)
ax.bar(X + 0.25, t1[0], color = '#54278f', width = 0.25, yerr = error1[2], bottom=0)

ax.bar(Y - 0.25, t2[2], color = '#FDE725', width = 0.25, yerr = error2[0], bottom=0)
ax.bar(Y + 0.00, t2[1], color = '#1F9E89', width = 0.25, yerr = error2[1], bottom=0)
ax.bar(Y + 0.25, t2[0], color = '#54278f', width = 0.25, yerr = error2[2], bottom=0)

ax.bar(Z - 0.25, t3[2], color = '#FDE725', width = 0.25, yerr = error3[0], bottom=0)
ax.bar(Z + 0.00, t3[1], color = '#1F9E89', width = 0.25, yerr = error3[1], bottom=0)
ax.bar(Z + 0.25, t3[0], color = '#54278f', width = 0.25, yerr = error3[2], bottom=0)

ax.bar(P - 0.25, t4[2], color = '#FDE725', width = 0.25, yerr = error4[0], bottom=0)
ax.bar(P + 0.00, t4[1], color = '#1F9E89', width = 0.25, yerr = error4[1], bottom=0)
ax.bar(P + 0.25, t4[0], color = '#54278f', width = 0.25, yerr = error4[2], bottom=0)

ax.bar(Q - 0.25, t5[2], color = '#FDE725', width = 0.25, yerr = error4[0], bottom=0)
ax.bar(Q + 0.00, t5[1], color = '#1F9E89', width = 0.25, yerr = error4[1], bottom=0)
ax.bar(Q + 0.25, t5[0], color = '#54278f', width = 0.25, yerr = error4[2], bottom=0)

ax.bar(R - 0.25, t6[2], color = '#FDE725', width = 0.25, yerr = error4[0], bottom=0)
ax.bar(R + 0.00, t6[1], color = '#1F9E89', width = 0.25, yerr = error4[1], bottom=0)
ax.bar(R + 0.25, t6[0], color = '#54278f', width = 0.25, yerr = error4[2], bottom=0)

ax.bar(A - 0.25, t7[2], color = '#FDE725', width = 0.25, yerr = error4[0], bottom=0)
ax.bar(A + 0.00, t7[1], color = '#1F9E89', width = 0.25, yerr = error4[1], bottom=0)
ax.bar(A + 0.25, t7[0], color = '#54278f', width = 0.25, yerr = error4[2], bottom=0)

ax.bar(B - 0.25, t8[2], color = '#FDE725', width = 0.25, yerr = error4[0], bottom=0)
ax.bar(B + 0.00, t8[1], color = '#1F9E89', width = 0.25, yerr = error4[1], bottom=0)
ax.bar(B + 0.25, t8[0], color = '#54278f', width = 0.25, yerr = error4[2], bottom=0)

# ax.bar(C - 0.25, t9[2], color = '#FDE725', width = 0.25, yerr = error4[0], bottom=0)
# ax.bar(C + 0.00, t9[1], color = '#1F9E89', width = 0.25, yerr = error4[1], bottom=0)
# ax.bar(C + 0.25, t9[0], color = '#54278f', width = 0.25, yerr = error4[2], bottom=0)


#ax.legend(labels=['concave', 'saddle', 'convex'], loc='upper left', fontsize=10)
ax.set(ylim=(0.0, 1.4))

# T = np.array([0,1,2,3,4,5,6,7,8])
# x_labels = ('GW29', 'GW31', 'GW33', 'GW35','GW37', 'GW39', 'GW41', 'GW43', 'Adult')
# plt.xticks(T, x_labels,rotation=90)


plt.savefig('barplot_grouped.jpg', dpi=500, bbox_inches='tight')


