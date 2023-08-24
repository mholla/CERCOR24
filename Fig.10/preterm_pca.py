#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:17:19 2023

@author: nagehan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pds
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

############################# READ DATA ##################################################
ct = [] 
CT = '/Users/.../preterm/preterm_ct.asc'
with open(CT,'r') as CT_file:
    CT_lines = CT_file.readlines() 
ct = np.zeros(len(CT_lines))  
for i in range(len(CT_lines)):
    CT_data = CT_lines[i].split()
    ct[i] = CT_data[0]
    
ctr = [] 
CTR = '/Users/.../preterm/preterm_ctr.asc'
with open(CTR,'r') as CTR_file:
    CTR_lines = CTR_file.readlines() 
ctr = np.zeros(len(CTR_lines))  
for i in range(len(CTR_lines)):
    CTR_data = CTR_lines[i].split()
    ctr[i] = CTR_data[0]
    
ct_gyr = [] 
CT_gyr = '/Users/.../preterm/preterm_ct_gyr.asc'
with open(CT_gyr,'r') as CT_gyr_file:
    CT_gyr_lines = CT_gyr_file.readlines() 
ct_gyr = np.zeros(len(CT_gyr_lines))  
for i in range(len(CT_gyr_lines)):
    CT_gyr_data = CT_gyr_lines[i].split()
    ct_gyr[i] = CT_gyr_data[0]
    
ct_sulc = [] 
CT = '/Users/.../preterm/preterm_ct_sulc.asc'
with open(CT,'r') as CT_file:
    CT_lines = CT_file.readlines() 
ct_sulc = np.zeros(len(CT_lines))  
for i in range(len(CT_lines)):
    CT_data = CT_lines[i].split()
    ct_sulc[i] = CT_data[0]
    
ct_sad = [] 
CT = '/Users/.../preterm/preterm_ct_sad.asc'
with open(CT,'r') as CT_file:
    CT_lines = CT_file.readlines() 
ct_sad = np.zeros(len(CT_lines))  
for i in range(len(CT_lines)):
    CT_data = CT_lines[i].split()
    ct_sad[i] = CT_data[0]
    
sa = [] 
SA = '/Users/.../preterm/preterm_sa.asc'
with open(SA,'r') as SA_file:
    SA_lines = SA_file.readlines() 
sa = np.zeros(len(SA_lines))  
for i in range(len(SA_lines)):
    SA_data = SA_lines[i].split()
    sa[i] = SA_data[0]
    
sa = sa/100

vol = [] 
VOL = '/Users/.../preterm/preterm_vol.asc'
with open(VOL,'r') as VOL_file:
    VOL_lines = VOL_file.readlines() 
vol = np.zeros(len(VOL_lines))  
for i in range(len(VOL_lines)):
    VOL_data = VOL_lines[i].split()
    vol[i] = VOL_data[0]
    
vol = vol/1000

sd = [] 
SD = '/Users/.../preterm/preterm_sd.asc'
with open(SD,'r') as SD_file:
    SD_lines = SD_file.readlines() 
sd = np.zeros(len(SD_lines))  
for i in range(len(SD_lines)):
    SD_data = SD_lines[i].split()
    sd[i] = SD_data[0]
    
gi = [] 
GI = '/Users/.../preterm/preterm_gi.asc'
with open(GI,'r') as GI_file:
    GI_lines = GI_file.readlines() 
gi = np.zeros(len(GI_lines))  
for i in range(len(GI_lines)):
    GI_data = GI_lines[i].split()
    gi[i] = GI_data[0]
    
si = [] 
SI = '/Users/.../preterm/preterm_si.asc'
with open(SI,'r') as SI_file:
    SI_lines = SI_file.readlines() 
si = np.zeros(len(SI_lines))  
for i in range(len(SI_lines)):
    SI_data = SI_lines[i].split()
    si[i] = SI_data[0]
    
age = [] #birth age
AGE = '/Users/.../preterm/preterm_age.asc'
with open(AGE,'r') as AGE_file:
    AGE_lines = AGE_file.readlines() 
age = np.zeros(len(AGE_lines))  
for i in range(len(AGE_lines)):
    AGE_data = AGE_lines[i].split()
    age[i] = AGE_data[0]

###############################################################################
agevsct = {"CT":ct, "SD":sd, "GI":gi, "CTR":ctr, "SA":sa,
           "CT_GYR":ct_gyr, "CT_SULC":ct_sulc, "SI":si, "VOL":vol,"AGE":age};
df = pds.DataFrame(data=agevsct);
print(df);

############################### OLS ###########################################
features = ["SD","GI",'VOL',"SI","CTR","CT","CT_GYR","CT_SULC","SA"]

#define predictor and response variables
x = df[features]
y = df['AGE']
# add constant
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())

################################## MANOVA #####################################
from dfply import *

df >> group_by(X.AGE) >> summarize(n=X['GI'].count(), mean=X['GI'].mean(), std=X['GI'].std())
df >> group_by(X.AGE) >> summarize(n=X['SA'].count(), mean=X['SA'].mean(), std=X['SA'].std())
df >> group_by(X.AGE) >> summarize(n=X['SD'].count(), mean=X['SD'].mean(), std=X['SD'].std())
df >> group_by(X.AGE) >> summarize(n=X['VOL'].count(), mean=X['VOL'].mean(), std=X['VOL'].std())
df >> group_by(X.AGE) >> summarize(n=X['CT'].count(), mean=X['CT'].mean(), std=X['CT'].std())
df >> group_by(X.AGE) >> summarize(n=X['CT_GYR'].count(), mean=X['CT_GYR'].mean(), std=X['CT_GYR'].std())
df >> group_by(X.AGE) >> summarize(n=X['CT_SULC'].count(), mean=X['CT_SULC'].mean(), std=X['CT_SULC'].std())
df >> group_by(X.AGE) >> summarize(n=X['SI'].count(), mean=X['SI'].mean(), std=X['SI'].std())
df >> group_by(X.AGE) >> summarize(n=X['CTR'].count(), mean=X['CTR'].mean(), std=X['CTR'].std())

from statsmodels.multivariate.manova import MANOVA
fit = MANOVA.from_formula('GI+SA+SD+SI+CT+CTR+VOL+CT_GYR+CT_SULC ~ AGE', data=df)
print(fit.mv_test())

################################################################################
# convert the continuous scan ages feature into categorical variable
# Reread all the variables! Lines 36 to 137!!!
for i in range(len(age)):
    if age[i] <= 31:
        age[i] = 1
    elif age[i] <= 37 and age[i] > 31:
        age[i] = 2
    else: 
        age[i] = 3
        
age = age.astype(str)
###############################################################################
agevsct = {"CT":ct, "SD":sd, "GI":gi, "CTR":ctr, "SA":sa,
           "CT_GYR":ct_gyr, "CT_SULC":ct_sulc, "SI":si, "VOL":vol,"AGE":age};
df = pds.DataFrame(data=agevsct);
print(df);

######################### STEPWISE REGRESSION #################################
features = ["GI",'VOL',"SA","CT","CTR","CT_GYR","CT_SULC","SI","SD"]

x = df[features]
y = df['AGE']

knn = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='kd_tree')

# Forward Selection
sfs1 = SFS(LinearRegression(), 
           forward=True,
           k_features=(3),
           floating=True,
           scoring='r2',
           cv=StratifiedKFold(5))

sfs1 = sfs1.fit(x, y)
sfs1.subsets_
sfs1.k_feature_idx_
sfs1.k_score_

table = pds.DataFrame.from_dict(sfs1.get_metric_dict()).T
fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')
# plt.ylim([0.8, 1])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()

# Backward Elimination
sfs2 = SFS(LinearRegression(), 
           forward=False,
           k_features=(3),
           floating=True,
           scoring='r2',
           cv=StratifiedKFold(5))

sfs2 = sfs2.fit(x, y)
sfs2.subsets_
sfs2.k_feature_idx_
sfs2.k_score_

table = pds.DataFrame.from_dict(sfs2.get_metric_dict()).T
fig2 = plot_sfs(sfs2.get_metric_dict(), kind='std_dev')
# plt.ylim([0.8, 1])
plt.title('Sequential Backward Selection (w. StdDev)')
plt.grid()
plt.show()

################################# PCA #############################################
features = ["GI",'VOL',"SA","CT","CTR","CT_GYR","CT_SULC","SI","SD"]
# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['AGE']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
loadings = pds.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)
loadings_matrix = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pds.DataFrame(loadings_matrix, columns=['PC1', 'PC2'], index=features)
principalDf = pds.DataFrame(data = principalComponents
             , columns = ['PC1','PC2'])
finalDf = pds.concat([principalDf, df[['AGE']]], axis = 1)
pca.explained_variance_ratio_
pca.explained_variance_
loading_matrix
cum_sum_eigenvalues = np.cumsum(pca.explained_variance_ratio_)

# Plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['1.0', '2.0', '3.0']
colors = ['g', '#08519c', '#440154']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['AGE'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
               , finalDf.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 25, alpha = 0.8)
#ax.legend(targets)
plt.savefig('preterm_pca.jpg', dpi=500, bbox_inches='tight')


# plt.bar(range(0,len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual explained variance')
# plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal component index')
# plt.legend()
# plt.tight_layout()
# plt.show()
