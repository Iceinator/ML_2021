# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:03:36 2021

@author: chris


"""
from Load_Data import *

import matplotlib.pyplot as plt
import seaborn as sns
df = df.drop(columns=['Oz ','doy'])


plt.figure(figsize=[12, 12])
sns.pairplot(df, kind="scatter")
plt.show()
plt.savefig("CorrPlot.png")

print("f")
