# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:35:54 2021

@author: chris
"""
# exercise 6.2.1
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
from toolbox_02450 import correlated_ttest
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from Load_Data import *

# Load data from matlab file

y = np.array([classDict[cl] for cl in classLabels])

N, M = X.shape
C = len(classNames)

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))

Error_test_LogReg = np.empty((K,1))
opt_lambdai = np.empty((K,1))


Error_test_NB = np.empty((K,1))
opt_alphai = np.empty((K,1))

Error_test_base = np.empty((K,1))

Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

j=0
for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    
    
    
    #Loob for logistisk regretion
    
    X1=X_train
    X2=y_train
    
    X_train_LogReg, X_test_LogReg, y_train_LogReg, y_test_LogReg = train_test_split(X_train, y_train, test_size=.80, stratify=y_train)
    # Try to change the test_size to e.g. 50 % and 99 % - how does that change the 
    # effect of regularization? How does differetn runs of  test_size=.99 compare 
    # to eachother?
    
    # Standardize the training and set set based on training set mean and std
    mu = np.mean(X_train_LogReg, 0)
    sigma = np.std(X_train_LogReg, 0)
    
    X_train_LogReg = (X_train_LogReg - mu) / sigma
    X_test_LogReg = (X_test_LogReg - mu) / sigma
    
    # Fit regularized logistic regression model to training data to predict 
    # the type of wine
    lambda_interval = np.logspace(-8, 2, 50)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    k=0
    for k in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty='l2',multi_class='auto', C=1/lambda_interval[k] )
        
        mdl.fit(X_train_LogReg, y_train_LogReg)
    
        y_train_est = mdl.predict(X_train_LogReg).T
        y_test_est = mdl.predict(X_test_LogReg).T
        
        train_error_rate[k] = np.sum(y_train_est != y_train_LogReg) / len(y_train_LogReg)
        test_error_rate[k] = np.sum(y_test_est != y_test_LogReg) / len(y_test_LogReg)
    
        w_est = mdl.coef_[0] 
        coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
    
    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]
    
    
    #Køre modellen på det ydre testsæt med den optimale lambda
    opt_lambdai[j] = opt_lambda
    mdl = LogisticRegression(penalty='l2',multi_class='auto', C=1/opt_lambda )
    mdl.fit(X_train, y_train)
    y_est_LogReg = mdl.predict(X_test).T
    Error_test_LogReg[j] = np.sum(y_est_LogReg != y_test)/len(y_test)
    
    
    
    
    
    #Baseline
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    y_est_base = dummy_clf.predict(y_test)
    
    Error_test_base[j] =1 - dummy_clf.score(y_est_base, y_test)
    
    
    
    
    
    #NB
    opt_alpha = np.zeros(10)
    Error_test_inter = np.zeros(10)
    X1 = OneHotEncoder().fit_transform(X=X)
    X_train = X1[train_index,:]
    
    X_test = X1[test_index,:]
    
    v=0
    CV = model_selection.KFold(n_splits=internal_cross_validation,shuffle=True)
    for train_index_NB, test_index_NB in CV.split(X_train):
        #print('Crossvalidation fold: {0}/{1}'.format(k+1,K))
    
        # extract training and test set for current CV fold
        X_train_NB = X_train[train_index_NB,:]
        y_train_NB = y_train[train_index_NB]
        X_test_NB = X_train[test_index_NB,:]
        y_test_NB = y_train[test_index_NB]
        
        alpha = np.linspace(0.1, 10,num=10) # pseudo-count, additive parameter (Laplace correction if 1.0 or Lidtstone smoothing otherwise)
        fit_prior = True   # uniform prior (change to True to estimate prior from data)
        
        # K-fold crossvalidation
        CV = model_selection.KFold(n_splits=internal_cross_validation,shuffle=True)
        
        #
        # We need to specify that the data is categorical.
        # MultinomialNB does not have this functionality, but we can achieve similar
        # results by doing a one-hot-encoding - the intermediate steps in in training
        # the classifier are off, but the final result is corrent.
        # If we didn't do the converstion MultinomialNB assumes that the numbers are
        # e.g. discrete counts of tokens. Without the encoding, the value 26 wouldn't
        # mean "the token 'z'", but it would mean 26 counts of some token,
        # resulting in 1 and 2 meaning a difference in one count of a given token as
        # opposed to the desired 'a' versus 'b'.
        #X_train_NB = OneHotEncoder().fit_transform(X=X_train_NB)
        
        errors = np.zeros(10)
        
        i=0
        for i in range(0,10):
            #print('Crossvalidation fold: {0}/{1}'.format(k+1,K))
        
            # extract training and test set for current CV fold
            
            a1 = np.asscalar(alpha[i])
            nb_classifier = MultinomialNB(alpha=a1,fit_prior=fit_prior)
            
            nb_classifier.fit(X_train_NB, y_train_NB)
            y_est_prob = nb_classifier.predict_proba(X_test_NB)
            y_est = np.argmax(y_est_prob,1)
            
            errors[i] = np.sum(y_est!=y_test_NB,dtype=float)/y_test_NB.shape[0]
            i+=1
            
        opt_alpha_idx = np.argmin(errors)
        opt_alpha[v] = alpha[opt_alpha_idx]
        Error_test_inter[v] = np.min(errors)
        
        
        v+=1
    opt_alphai_idx = np.argmin(Error_test_inter)
    opt_alphai[j] = opt_alpha[opt_alphai_idx] 
        
    a1 = np.asscalar(opt_alphai[j])
    nb_classifier = MultinomialNB(alpha=a1,fit_prior=fit_prior)
    nb_classifier.fit(X_train, y_train)
    y_est_prob_NB = nb_classifier.predict_proba(X_test)
    y_est = np.argmax(y_est_prob_NB,1)
    Error_test_NB[j] = np.sum(y_est != y_test)/len(y_test)
    print('HEj')
    j+=1
    
   
    

#%%
# stats
r = []
r_log_NB = Error_test_LogReg - Error_test_NB
r_log_base = Error_test_LogReg - Error_test_base
r_NB_base = Error_test_NB - Error_test_base
alpha = 0.05
rho = 1/K
p_setupII_log_NB, CI_setupII_log_NB = correlated_ttest(r_log_NB, rho, alpha=alpha)
p_setupII_log_base, CI_setupII_log_base = correlated_ttest(r_log_base, rho, alpha=alpha)
p_setupII_NB_base, CI_setupII_NB_base = correlated_ttest(r_NB_base, rho, alpha=alpha)
print( [p_setupII_log_NB] )
print(CI_setupII_log_NB)


#%%
fig = plt.figure()
plt.figure(figsize=(15,10))


plt.plot(Error_test_LogReg, label = 'LogReg')
plt.plot(Error_test_NB, label = 'NB')
plt.plot(Error_test_base, ':b',label = 'Baseline')
plt.xlabel("Folds")
plt.ylabel("Error");
plt.legend();