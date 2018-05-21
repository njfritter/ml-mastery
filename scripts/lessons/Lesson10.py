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
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, RandomizedSearchCV
# Define url and columns
url = 'https://goo.gl/bDdBiA'
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Read in data
data = read_csv(url, names = columns)
array = data.values

####################################
# Lesson 10: Improve Accuracy with Algorithm Tuning
####################################

'''
Now that we have model candidates that have drawn mostly even in our comparisons,
Let's do some algorithm tuning to see how high our accuracy can go.
This process involves looking at the different algorithm parameters 
and trying to find the combination that yields the best model 
(in terms of the metrics you are analyzing).

Here you can find the different models and their parameters:
http://scikit-learn.org/stable/modules/classes.html

For example, the Gradient Boosted Machine we will be training has these parameters:
- loss: Loss function that we will to minimize
- learning_rate: Rate at which contribution of each tree shrinks
- n_estimators: Number of boosting stages (iterative tree model)
- max_depth: maximum number of nodes of each tree

There are more, but these are the main ones I will be tuning.

'''

# Divide data into attributes and predictor
X = array[:, 0:8]
y = array[:, 8]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

# Now one at a time, I will be declaring the model parameters to tune
# And then running a Grid Search to determine the best combination
# Will also implement a Randomized Grid Search to compare
# Which will take a random subset of the parameters given

# NOTE: The Logistic Regression model does not have many parameters
# to optimize, as it is a simpler model. Won't be trying too many combos

# Parameters for Logistic Regression
grid_param_lr = {
	'penalty': ['l1', 'l2'],
	'class_weight': ['balanced', None],
	'C': (0.5, 0.75, 1, 1.25, 1.5)
}

# First try Logistic Regression model
model_lr = LogisticRegression(random_state = 1, verbose = 5)

grid_lr = GridSearchCV(estimator = model_lr, param_grid = grid_param_lr)
grid_lr.fit(X, y)
#print(grid_lr.best_score_)
#print(grid_lr.best_estimator_)
print(grid_lr.cv_results_)

random_lr = RandomizedSearchCV(estimator = model_lr, param_distributions = grid_param_lr)
random_lr.fit(X, y)
#print(random_lr.best_score_)
#print(random_lr.best_estimator_)
print(random_lr.cv_results_)

# Will try the other methods later

"""
models[1] = ['Linear Discriminant Analysis', LinearDiscriminantAnalysis()]	
models[2] = ['Gradient Boosted Machine', GradientBoostingClassifier(random_state = 1)]
"""