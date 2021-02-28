# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 13:44:06 2021

@author: Nicklas Rasmussen
"""
import numpy as np
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
from pathlib import Path

path = Path(__file__).parent / "../Data/LA_Ozone_metric.csv"
with path.open() as f:
    df = pd.read_csv(f)

#% Load data
raw_data = df.values

cols = range(0, 10) 
X = np.array(list(raw_data[:, cols]),dtype=np.float)
attributeNames = np.asarray(df.columns[cols])


#% Catergoizing concentration of Ozone as high, medium and low
# Beregner de 3 kvartiler for at inddele 
ozone = X[:,0]
median = st.median(ozone) # = 10
avg = st.mean(ozone) # = 11.78
limit_low = min(ozone) # = 1
limit_high = max(ozone) # 38
p1 = np.percentile(ozone,30) # 6
p2 = np.percentile(ozone,70) # 14

# Overview of data
plt.boxplot(ozone)
plt.show()

plt.hist(ozone)
plt.show()

# Choses boundary to be represented by the 33'rd and 66'th percentiles. 
category = ['Low' if i <=p1 else 'High' if i >=p2 else 'Medium' for i in ozone]

# Assigning values for corresponding categories.
classLabels = category 

# Then determine which classes are in the data by finding the set of 
# unique class labels 
classNames = np.unique(classLabels)

# We can assign each type of Iris class with a number by making a
# Python dictionary as so:

classes = [3,1,2]
classDict = dict(zip(classNames,classes))
# The function zip simply "zips" togetter the classNames with an integer,
# like a zipper on a jacket. 
# For instance, you could zip a list ['A', 'B', 'C'] with ['D', 'E', 'F'] to
# get the pairs ('A','D'), ('B', 'E'), and ('C', 'F'). 
# A Python dictionary is a data object that stores pairs of a key with a value. 
# This means that when you call a dictionary with a given key, you 
# get the stored corresponding value. Try highlighting classDict and press F9.
# You'll see that the first (key, value)-pair is ('Iris-setosa', 0). 
# If you look up in the dictionary classDict with the value 'Iris-setosa', 
# you will get the value 0. Try it with classDict['Iris-setosa']

# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is 
# assigned. This is the class index vector y:
y = np.array([classDict[cl] for cl in classLabels])

# Adds this to the original dataset
N = len(X)
X = np.c_[np.zeros(N),X]
X[:,0] = y

# In the above, we have used the concept of "list comprehension", which
# is a compact way of performing some operations on a list or array.
# You could read the line  "For each class label (cl) in the array of 
# class labels (classLabels), use the class label (cl) as the key and look up
# in the class dictionary (classDict). Store the result for each class label
# as an element in a list (because of the brackets []). Finally, convert the 
# list to a numpy array". 

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)

