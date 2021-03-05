# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:38:32 2021

@author: Nicklas Rasmussen
"""
#%% Categorization

from Load_Data import *

#% Catergorizing concentration of Ozone as high, medium and low
# Beregner de 3 kvartiler for at inddele 
ozone = y
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

classes = [3,1,2]
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,classes))


# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is 
# assigned. This is the class index vector y:
y = np.array([classDict[cl] for cl in classLabels])

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)



