#!/usr/bin/env python3

###################################
# Mastering ML Python Mini Course
#
# Inspired by the project here: 
#
# https://s3.amazonaws.com/MLMastery/machine_learning_mastery_with_python_mini_course.pdf?__s=mxhvphowryg2sfmzus2q
#
# By Nathan Fritter
#
# Project will soon be found at: 
#
# https://www.inertia7.com/projects/
####################################

# Welcome to my repo for the Mastering Machine Learning Python Mini Course
# Here I will be going through each part of the course
# So you can get a feel of the different parts

import numpy as np
import pandas as pd
from pandas import read_csv, Series
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix
# Define url and columns
url = 'https://goo.gl/bDdBiA'
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Read in data
data = read_csv(url, names = columns)
array = data.values

####################################
# Lesson 8: Algorithm Evaluation Metrics
####################################

'''
This was originally supposed to come before the model spot checking
But I feel like different model evlauation metrics are important
After one has narrowed down the potential choice of models

The candidates for further analysis from the previous Lesson:
- Logistic Regression
- Linear Discriminant Analysis
- Random Forest
- Gradient Boosted Machine

In this I will utilize k-fold cross validation 
in tandem with different model metrics and evaluation methods.
As some model metrics may not tell the whole story
(i.e. accuracy not detailing False Positive vs True Positive, etc.)
'''

# Divide data into attributes and predictor
X = array[:, 0:8]
y = array[:, 8]

# Make list for models
models = np.empty([4, 2], dtype = object)

# Linear models
models[0] = ['Logistic Regression', LogisticRegression(random_state = 1)]
models[1] = ['Linear Discriminant Analysis', LinearDiscriminantAnalysis()]

# More complex models	
models[2] = ['Random Forest', RandomForestClassifier(random_state = 1)]
models[3] = ['Gradient Boosted Machine', GradientBoostingClassifier(random_state = 1)]


# Iterate through models, then fit & evaluate 
for name, model in models:

    k_fold = KFold(n_splits = 10, random_state = 1)	
    for scoring in ('accuracy', 'roc_auc', 'neg_log_loss'):
        try:
            result = cross_val_score(model, X, y, cv = k_fold, scoring = scoring)
            if scoring == 'accuracy':
                print("\n%s of %s model:\n %.3f%% (+\-%.3f%%)" % 
                (scoring, name, result.mean() * 100.0, result.std() * 100.0))
            else:
                print("\n%s of %s model:\n %.3f (+\-%.3f)" % 
                (scoring, name, result.mean(), result.std()))
        except AttributeError:
            print("The %s model cannot perform cross validation with the %s metric" % (name, scoring))


'''
Here it looks like we get a bit of separation. Results:

Logistic Regression accuracy: 76.951% (+\-4.841%)
Logistic Regression AUC: 0.823 (+\-0.041)
Logistic Regression LogLoss: -0.493 (+\-0.047)

Linear Discriminant Analysis accuracy: 77.346% (+\-5.159%)
Linear Discriminant Analysis AUC: 0.829 (+\-0.044)
Linear Discriminant Analysis LogLoss: -0.486 (+\-0.064)

Random Forest model accuracy: 74.077% (+\-5.791%)
Random Forest model AUC: 0.791 (+\-0.037)
Random Forest model LogLoss: -1.219 (+\-0.391)

Gradient Boosted Machine accuracy: 76.950% (+\-5.710%)
Gradient Boosted Machine AUC: 0.829 (+\-0.047)
Gradient Boosted Machine LogLoss:-0.483 (+\-0.086)


The random forest consistently underperforms in all three metrics,
so that one will be eliminated from the running. The three remaining:
- Logistic Regression
- Linear Discriminant Analysis
- Gradient Boosted Machine
'''