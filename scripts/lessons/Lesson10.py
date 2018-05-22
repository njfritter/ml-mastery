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

# Divide data into attributes and predictor
X = array[:, 0:8]
y = array[:, 8]

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
Other algorithms may have more or less to tune; the simpler the model,
the less parameters there are to tune.

The golden rule here is Occam's Razor:
The hypothesis (and in our case, model) with the least number of assumptions
(and again in our case, model parameters) is usually correct
'''

# Define a function to do parameter tuning
# Since we will be doing it multiple times
def parameter_tuning(model, parameters, attributes, labels):
    grid_model = GridSearchCV(estimator = model, param_grid = parameters)
    grid_model.fit(attributes, labels)
    
    #print("\nCross validation results for Grid Search", grid_model.cv_results_)
    print("\nBest score in Grid Search:\n", grid_model.best_score_)
    print("\nBest set of parameters in Grid Search:\n", grid_model.best_estimator_)
    
    rand_model = RandomizedSearchCV(estimator = model, param_distributions = parameters)
    rand_model.fit(attributes, labels)
    
    #print("\nCross validation results", rand_model.cv_results_)
    print("\nBest score in Randomized Search:\n", rand_model.best_score_)
    print("\nBest set of parameters in Randomized Search:\n", rand_model.best_estimator_)

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
model_lr = LogisticRegression(random_state = 1)
parameter_tuning(model_lr, grid_param_lr, X, y)

# Parameters for Linear Discriminant Analysis
grid_param_lda = {
	'solver': ['svd', 'lsqr', 'eigen'],
	'shrinkage': [None, 'auto', 0.25, 0.5, 0.75, 1]
}

# Now Linear Discriminant Analysis
model_lda = LinearDiscriminantAnalysis()
parameter_tuning(model_lda, grid_param_lda, X, y)

# Parameters for Gradient Boosted Machine
grid_param_gbm = {
	'loss': ['deviance', 'exponential'],
	'learning_rate': (0.5, 0.1, 0.05, 0.01),
	'n_estimators': (50, 100, 250, 500),
	'max_depth': (2, 3, 4, 5)
}

# And lastly Gradient Boosted Machine
model_gbm = GradientBoostingClassifier(random_state = 1)
parameter_tuning(model_gbm, grid_param_gbm, X, y)