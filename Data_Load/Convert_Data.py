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


#% Load data
raw_data = df.values

cols = range(0, 10) 
X = np.array(list(raw_data[:, cols]),dtype=np.float)
#%
# As many of the units in the data is given in the imperial system, the header is first changed to also include units. 
# Later, the data is converted to the metric system
attributeNames = np.asarray(df.columns[cols])
attributeNames = ['Ozone [ppm]','Vandenberg height [m]', 'Wind speed [m/s]','Humidity [%]','Temperature [C]','Inverse Base Temperaure [C]','Dpg [mm Hg]','Inverse Base Height [m]','Visibility [m]','doy [days]']

# Convert the variables to the metric system.

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


#%% The data is now converted. Adds the attributeNames and saves as a new csv-file
df = pd.DataFrame(X, columns = attributeNames)

pd.DataFrame(df).to_csv('C:/Users/Nicklas Rasmussen/Desktop/DTU/6.semester/02450_Intro to Macine Learning and Data Mining/Projekt1/ML_2021/Data/LA_Ozone_metric.csv',index=False)

