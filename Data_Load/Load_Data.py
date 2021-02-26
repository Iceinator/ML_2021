# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:49:38 2021

@author: Nicklas Rasmussen
"""

import numpy as np
import pandas as pd
from pathlib import Path

path = Path(__file__).parent / "../Data_Load/LA_Ozone.csv"
with path.open() as f:
    df = pd.read_csv(f)

#%% Load data

# Load the Iris csv data using the Pandas library
#df = pd.read_csv(f)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise

raw_data = df.values
# Notice that raw_data both contains the information we want to store in an array
# X (the sepal and petal dimensions) and the information that we wish to store 
# in y (the class labels, that is the iris species).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.
cols = range(0, 10) 
X = np.array(list(raw_data[:, cols]),dtype=np.float)

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])
attributeNames = ['Ozone [ppm]','Vandenberg height [m]', 'Wind speed [m/s]','Humidity [%]','Temperature [C]','Inverse Base Temperaure [C]','Dpg [mm Hg]','Inverse Base Height [m]','Visibility [m]','doy [days]']
# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape

# Have to convert the variables to the metric system.

# Temperature (from Fahrenheit to Celcius)
# Rounds op to nearest integer, as the data frame X 
# automatically rounds up (and does not do it right by default)
Celcius_t = np.round((X[:,4]-32)*(5/9),1)
X[:,4] = Celcius_t

# Wind speed (from mph to m/s)
windspeed = np.round(X[:,2]*0.44704,1)
X[:,2] = windspeed

# Visibility (from miles to meters in order to minimize
# the magnitude of the rounding)
visibility = np.round(X[:,8]*1609.34,1)
X[:,8] = visibility

# Inversion Base Height (from feet to m)
ibh = np.round(X[:,5]*0.3048,1)
X[:,5] = ibh

# Inversion base temperature (from Fahrenheit to Celcius)
Celcius_ibt = np.round((X[:,7]-32)*(5/9),1)
X[:,7] = Celcius_ibt

metric_data = np.vstack((attributeNames,X) )
pd.DataFrame(X).to_csv('LA_Ozone_metric.csv')
#np.savetxt("LA_Ozone_metric.csv",X,delimiter=",")

#%% Vi kan overveje at opdele i tyk, mellem og tynd - hvis ikke, så er denne ligegyldig
# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)

#%% Identificér brugbar/ikke brugbar data
