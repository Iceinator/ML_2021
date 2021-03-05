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

#import os
#os.chdir('C:/Users/Nicklas Rasmussen/Desktop/DTU/6.semester/02450_Intro to Macine Learning and Data Mining/Projekt1/ML_2021/Data_Load')

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