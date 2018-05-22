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
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
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
# Lesson 11: Improve Accuracy with Ensemble Methods
####################################

'''
Here in the course would have been a section to do some ensemble model 
training, as it represents an extra layer on top of traditional models
But since I have already done this, 
I will instead invoke the one ensemble method I haven't tried:
The Voting Classifier

This method involves literally combining different models 
(such as Logsitic Regression + Decision Tree) versus many of the same models
(many Decision Trees in a Random Forest or Gradient Boosted Machine)

Here I will try out a bunch of different things and see where it goes!
Will use cross validation metrics here, nothing too fancy
'''

# Make list for models
models = np.empty([3, 2], dtype = object)

# Voting ensembles
# Number 1: Hard Vote (Predicted class labels used for majority rule voting)
models[0] = ['Voting Classifier 1', VotingClassifier(estimators = [
    ('lr', LogisticRegression(random_state = 1)),
    ('gbm', GradientBoostingClassifier(random_state = 1)),],
    voting = 'hard')]

# Number 2: Soft Vote (Argmax of sums of predicted probabilities used)
# Recommended for ensemble of well-calibrated classifiers
models[1] = ['Voting Classifier 2', VotingClassifier(estimators = [
    ('lda', LinearDiscriminantAnalysis()),
    ('lr', LogisticRegression(random_state = 1))],
    voting = 'soft')]

# Number 3: Soft Vote with weights
# Some models will be more valuable than others
models[2] = ['Voting Classifier 3', VotingClassifier(estimators = [
    ('lr', LogisticRegression(random_state = 1)),
    ('gbm', GradientBoostingClassifier(random_state = 1)),],
    voting = 'soft',
    weights = (0.25, 0.75))]

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

