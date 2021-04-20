# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:47:11 2021

@author: chris
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

from toolbox_02450 import rocplot, confmatplot
from Load_Data import *

#%%
font_size = 15
plt.rcParams.update({'font.size': font_size})

# Load Matlab data file and extract variables of interest

y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape
C = len(classNames)


#%%

dummy_clf = DummyClassifier(strategy="most_frequent")
K = dummy_clf.fit(X, y)

HEj = dummy_clf.predict(y)

Hej2 = 1 - dummy_clf.score(HEj, y)
