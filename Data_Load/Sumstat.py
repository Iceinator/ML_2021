# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:31:21 2021

@author: chris

Summery statistics
"""
from Load_Data import *



gg = df.describe()



pd.DataFrame(gg).to_csv("../Data_Load/Sumstat.csv",index=False)