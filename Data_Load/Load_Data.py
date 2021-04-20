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
from scipy.linalg import svd

path = Path(__file__).parent / "../Data/LA_Ozone_metric.csv"
with path.open() as f:
    df = pd.read_csv(f)

# % Standard-form
#% Load data
raw_data = df.values

cols = range(0, 10) 
X = np.array(list(raw_data[:, cols]),dtype=np.float)
attributeNames = np.asarray(df.columns[cols])
# Remove last and first column of attribute name (ozone and day of year)
attributeNames = attributeNames[1:]
attributeNames = attributeNames[:-1]

y = X[:,0]

# Removes the y-column from the X-matrix and the "day of the year"-column
X = X[:,1:]
X = np.delete(X,-1,1)

N = len(y)
# Har fjernet 2 attributter
M = len(attributeNames);

ozone = y
median = st.median(ozone) # = 10
avg = st.mean(ozone) # = 11.78
limit_low = min(ozone) # = 1
limit_high = max(ozone) # 38
p1 = np.percentile(ozone,30) # 6
p2 = np.percentile(ozone,70) # 14
# Choses boundary to be represented by the 33'rd and 66'th percentiles. 
category = ['Low' if i <=p1 else 'High' if i >=p2 else 'Medium' for i in ozone]

# Assigning values for corresponding categories.
classLabels = category 

# Then determine which classes are in the data by finding the set of 
# unique class labels 
classNames = np.unique(classLabels)
# 201
classes = [0,1,2]
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,classes))