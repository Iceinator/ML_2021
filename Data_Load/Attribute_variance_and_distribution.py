# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:09:54 2021

@author: Nicklas Rasmussen
"""

from matplotlib.pyplot import (figure, title, boxplot, xticks, subplot, hist,
                               xlabel, ylim, yticks, show)
from scipy.io import loadmat
from scipy.stats import zscore
from Load_Data import *

# By looking at the variable X, it is clear that the magnitude
# of the attributes differ a lot. Thus, a histogram is made of 
# each attribute alongside a boxplot in order to detect outliers


# Boxplot of all attributes
figure()
title('Ozone: Boxplot')
boxplot(X)
xticks(range(1,M+1), attributeNames, rotation=45)

# Standardized boxplot of all attributes
figure(figsize=(12,6))
title('Wine: Boxplot (standarized)')
boxplot(zscore(X, ddof=1))
xticks(range(1,M+1), attributeNames, rotation=45)

# Histogram for all attributes
figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); 
v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: yticks([])
    if i==0: title('Ozone: Histogram')

# Historgram with outlier attributes
figure(figsize=(14,9))
m = [0, 1, -1]
for i in range(len(m)):
    subplot(1,len(m),i+1)
    hist(X[:,m[i]],50)
    xlabel(attributeNames[m[i]])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i>0: yticks([])
    if i==0: title('Ozone: Histogram (Vandenberg Height, Wind speed & Visibility)')

# No concrete outliers

