#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:10:19 2023

@author: nagehan
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sage = [] 
SAGE = '/Users/.../scan_ages.txt'
with open(SAGE,'r') as SAGE_file:
    SAGE_lines = SAGE_file.readlines() 
sage = np.zeros(len(SAGE_lines))  
for i in range(len(SAGE_lines)):
    SAGE_data = SAGE_lines[i].split()
    sage[i] = SAGE_data[0]      
        
bage = [] 
BAGE = '/Users/.../birth_ages.txt'
with open(BAGE,'r') as BAGE_file:
    BAGE_lines = BAGE_file.readlines() 
bage = np.zeros(len(BAGE_lines))  
for i in range(len(BAGE_lines)):
    BAGE_data = BAGE_lines[i].split()
    bage[i] = BAGE_data[0]
    
sage=np.round(sage)   
bage=np.round(bage)  
sns.histplot(bage, bins=15, discrete=True, color='gray', label='birth age')
sns.histplot(sage, bins=15, discrete=True, color='lightgray', label='scan age')
x = [25, 30, 35, 40]
plt.xticks(x)
plt.legend()
plt.savefig('ages.jpg', dpi=500, bbox_inches='tight')