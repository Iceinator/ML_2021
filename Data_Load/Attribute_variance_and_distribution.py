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
title('Ozone: Boxplot of attributes')
boxplot(X)
xticks(range(1,M+1), attributeNames, rotation=45)

# Standardized boxplot of all attributes
figure(figsize=(12,6))
title('Ozone: Boxplot of standardized attributes')
boxplot(zscore(X, ddof=1))
xticks(range(1,M+1), attributeNames, rotation=45)

# Histogram for all attributes
figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); 
v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i],edgecolor='black')
    xlabel(attributeNames[i])
    ylim(0, 150) # Make the y-axes equal for improved readability

    if i%v!=0: yticks([])
    if i==0: title('Ozone: Histogram of attributes')

# Historgram with outlier attributes
figure(figsize=(14,9))
m = [0, 1, -1]
for i in range(len(m)):
    subplot(1,len(m),i+1)
    hist(X[:,m[i]],50,edgecolor='black')
    xlabel(attributeNames[m[i]])
    ylim(0, 100) # Make the y-axes equal for improved readability
    if i>0: yticks([])
    if i==0: title('Ozone: Histogram of attributes with outliers (VH, WS and Vi)')

# No concrete outliers




#%% Correlation
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show)

# requires data from exercise 4.2.1
from Load_Data import *

figure(figsize=(12,10))
for m1 in range(M):
    for m2 in range(M):
        subplot(M, M, m1*M + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X[class_mask,m2]), np.array(X[class_mask,m1]), '.')
            if m1==M-1:
                xlabel(attributeNames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(attributeNames[m1])
            else:
                yticks([])
        
legend(classNames)

show()

print('Ran Exercise 4.2.5')



#%% Correlation
Correlation = np.corrcoef(X)


