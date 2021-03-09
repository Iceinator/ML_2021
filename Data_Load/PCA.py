# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:05:25 2021

@author: Nicklas Rasmussen
"""

# From exercise 2.1.3
from Load_Data import *
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
#% Categorization
# Catergorizing concentration of Ozone as high, medium and low
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
plt.title('Boxplot of Ozone Concentration')
plt.xticks(range(1))
plt.ylabel('Ozone [ppm]')
plt.show()
plt.hist(ozone,edgecolor='black')
plt.title('Histogram of Ozone Concentration')
xlabel('Ozone [ppm]')
ylabel('Frequency')
plt.show()

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

#%
# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is 
# assigned. This is the class index vector y:
y = np.array([classDict[cl] for cl in classLabels])

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)




# Subtract mean value from data (mean gøres til en matrix for at dimensioner passer)
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.90

# Plot variance explained
plt.figure()

plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')

plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()
#print('Ran Exercise 2.1.3')


#%% Standardized data projected onto PC i and j 

Y = X - np.ones((N,1))*X.mean(axis=0)
Y = Y*(1/np.std(X,0))

U,S,Vh = svd(Y,full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 1
j = 0

# Plot PCA of the data
f = figure()
title('Ozone data: PC1 vs PC2')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()


# Indices of the principal components to be plotted
i = 0
j = 2

# Plot PCA of the data
f = figure()
title('Ozone data: PC1 vs PC3')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()


# Indices of the principal components to be plotted
i = 0
j = 3

# Plot PCA of the data
f = figure()
title('Ozone data: PC2 vs PC3')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
    #plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()


# Indices of the principal components to be plotted
i = 0
j = 4

# Plot PCA of the data
f = figure()
title('Ozone data: PC2 vs PC3')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
    #plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()



# Indices of the principal components to be plotted
i = 1
j = 2

# Plot PCA of the data
f = figure()
title('Ozone data: PC2 vs PC3')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
    #plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()


# Indices of the principal components to be plotted
i = 1
j = 3

# Plot PCA of the data
f = figure()
title('Ozone data: PC2 vs PC3')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
    #plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()

# Indices of the principal components to be plotted
i = 1
j = 4

# Plot PCA of the data
f = figure()
title('Ozone data: PC2 vs PC3')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
    #plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()



print('Ran Exercise 2.1.4')



#%%
# Outcomment all "single" #'s in order to do standardized data
#U,S,Vh = svd(Y,full_matrices=False)
#V = Vh.T    '
## We saw in 2.1.3 that the first 3 components explaiend more than 90
## percent of the variance. Let's look at their coefficients:

    
pcs = [0,1,2,3,4]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
plt.figure(22)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw,edgecolor="black")
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Ozone: PCA Component Coefficients')
plt.show()

# Inspecting the plot, we see that the 2nd principal component has large
# (in magnitude) coefficients for attributes A, E and H. We can confirm
# this by looking at it's numerical values directly, too:
print('PC2:')
print(V[:,1].T)

# How does this translate to the actual data and its projections?
# Looking at the data for water:

# Projection of water class onto the 2nd principal component.
high = Y[y==0,:]

print('First high observation')
print(high[0,:])

# Based on the coefficients and the attribute values for the observation
# displayed, would you expect the projection onto PC2 to be positive or
# negative - why? Consider *both* the magnitude and sign of *both* the
# coefficient and the attribute!

# You can determine the projection by (remove comments):
print('...and its projection onto PC2')
print(high[0,:]@V[:,1])
# Try to explain why?

# Exercise  2.1.5

#%%
## exercise 2.1.6
r = np.arange(1,X.shape[1]+1)
plt.bar(r, np.std(X,axis=0),edgecolor = 'black')
plt.xticks(r, attributeNames)
plt.ylabel('Standard deviation')
plt.xlabel('Attributes')
plt.title('Ozone: attribute standard deviations')
#%
## Investigate how standardization affects PCA

# Try this *later* (for last), and explain the effect
#X_s = X.copy() # Make a to be "scaled" version of X
#X_s[:, 2] = 100*X_s[:, 2] # Scale/multiply attribute C with a factor 100
# Use X_s instead of X to in the script below to see the difference.
# Does it affect the two columns in the plot equally?


# Subtract the mean from the data
Y1 = (X - np.ones((N, 1))*X.mean(0))

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y2 = X - np.ones((N, 1))*X.mean(0)
Y2 = Y2*(1/np.std(Y2,0))
# Here were utilizing the broadcasting of a row vector to fit the dimensions 
# of Y2

# Store the two in a cell, so we can just loop over them:
Ys = [Y1, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=.4)
plt.title('NanoNose: Effect of standardization')
nrows=3
ncols=2
for k in range(2):
    # Obtain the PCA solution by calculate the SVD of either Y1 or Y2
    U,S,Vh = svd(Ys[k],full_matrices=False)
    V=Vh.T # For the direction of V to fit the convention in the course we transpose
    # For visualization purposes, we flip the directionality of the
    # principal directions such that the directions match for Y1 and Y2.
    if k==1: V = -V; U = -U; 
    
    # Compute variance explained
    rho = (S*S) / (S*S).sum() 
    
    # Compute the projection onto the principal components
    Z = U*S;
    
    # Plot projection
    plt.subplot(nrows, ncols, 1+k)
    C = len(classNames)
    for c in range(C):
        plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
    plt.xlabel('PC'+str(i+1))
    plt.xlabel('PC'+str(j+1))
    plt.title(titles[k] + '\n' + 'Projection' )
    plt.legend(classNames)
    plt.axis('equal')
    
    # Plot attribute coefficients in principal component space
    plt.subplot(nrows, ncols,  3+k)
    for att in range(V.shape[1]):
        plt.arrow(0,0, V[att,i], V[att,j])
        plt.text(V[att,i], V[att,j], attributeNames[att])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
    plt.title(titles[k] +'\n'+'Attribute coefficients')
    plt.axis('equal')
            
    # Plot cumulative variance explained
    plt.subplot(nrows, ncols,  5+k);
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.title(titles[k]+'\n'+'Variance explained')

plt.show()
        
