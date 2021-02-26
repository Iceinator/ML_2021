# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:49:38 2021

@author: Nicklas Rasmussen
"""

import numpy as np
import pandas as pd
from pathlib import Path
import csv

path = Path(__file__).parent / "../Data_Load/LA_Ozone.csv"
with path.open() as f:
    test = list(csv.reader(f))

#%% Load data

# Load the Iris csv data using the Pandas library
filename = 'C:/Users/Nicklas Rasmussen/Desktop/DTU/6.semester/02450_Intro to Macine Learning and Data Mining/LA_Ozone.csv'
df = pd.read_csv(filename)

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
X = raw_data[:, cols]

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape


# Bemærk, alle data-punkter er diskrete, den sidste kolonne angiver blot hvilken dag der er tale om.

#%% Vi kan overveje at opdele i tyk, mellem og tynd - hvis ikke, så er denne ligegyldig
# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)

#%% Identificér brugbar/ikke brugbar data
