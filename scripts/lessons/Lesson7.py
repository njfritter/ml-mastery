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
# Define url and columns
url = 'https://goo.gl/bDdBiA'
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Read in data
data = read_csv(url, names = columns)
array = data.values

####################################
# Lesson 7: Algorithm Evaluation With Resampling Methods
####################################

'''
This was originally supposed to come before the model spot checking
But I feel like resampling methods and cross validation are more appropriate
After one has narrowed down the potential choice of models

The candidates for further analysis from the previous Lesson:
- Logistic Regression
- Linear Discriminant Analysis
- Random Forest
- Gradient Boosted Machine

In this I will utilize k-fold cross validation to sift through the remaining algorithms
Will; still use accuracy as the main evaluation metric
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
    try:
        result = cross_val_score(model, X, y, cv = k_fold, scoring = 'accuracy')
        print("\nAccuracy of %s model:\n %.3f%% (+\-%.3f%%)" % 
        (name, result.mean() * 100.0, result.std() * 100.0))
    except AttributeError:
        print("The %s model cannot perform cross validation with the %s metric" % (name, scoring))


'''
Looks like all of the models continue to perform well. Results:

- Logistic Regression: 76.951% (+\-4.841%)
- Linear Discriminant Analysis: 77.346% (+\-5.159%)
- Random Forest: 74.077% (+\-5.791%)
- Gradient Boosted Machine: 76.950% (+\-5.710%)

Thus they all move onto the next stage, which will look at more model metrics and evaluation
'''