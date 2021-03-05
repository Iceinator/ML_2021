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



#%%
bins = 20

plt.subplot(1,2,1)
plt.hist(X[:,0],bins)
plt.title('Vandenberg height')
plt.xlabel('Height [m]')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
plt.boxplot(X[:,0])
plt.title('Boxplot of Vandenberg Height')

plt.show()



plt.subplot(1,2,1)
plt.hist(X[:,1],bins)
plt.title('Wind speed')
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
plt.boxplot(X[:,1])
plt.title('Boxplot of Wind speed')

plt.show()



plt.subplot(1,2,1)
plt.hist(X[:,2],bins)
plt.title('Humidity')
plt.xlabel('Humidity [%]')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
plt.boxplot(X[:,2])
plt.title('Boxplot of Humidity')

plt.show()



plt.subplot(1,2,1)
plt.hist(X[:,3],bins)
plt.title('Temperature')
plt.xlabel('Temperature [C]')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
plt.boxplot(X[:,3])
plt.title('Boxplot of Temperature')

plt.show()


plt.subplot(1,2,1)
plt.hist(X[:,4],bins)
plt.title('Inverse Base Temperature [C]')
plt.xlabel('Tempearture [C]')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
plt.boxplot(X[:,4])
plt.title('Boxplot of IBT')

plt.show()


plt.subplot(1,2,1)
plt.hist(X[:,5],bins)
plt.title('Dpg')
plt.xlabel('Dpg [mmHg]')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
plt.boxplot(X[:,5])
plt.title('Boxplot of Dpg')

plt.show()


plt.subplot(1,2,1)
plt.hist(X[:,6],bins)
plt.title('Inverse Base Height')
plt.xlabel('Height [m]')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
plt.boxplot(X[:,6])
plt.title('Boxplot of IBH')

plt.show()


plt.subplot(1,2,1)
plt.hist(X[:,7],bins)
plt.title('Visibility')
plt.xlabel('Distance [km]')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
plt.boxplot(X[:,7])
plt.title('Boxplot of Dpg')

plt.show()

# It is clear that the "Visibility" attribute