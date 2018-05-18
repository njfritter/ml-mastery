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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Define url and columns
url = 'https://goo.gl/bDdBiA'
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Read in data
data = read_csv(url, names = columns)
array = data.values

####################################
# Lesson 6: Spot Check Algorithms
####################################

'''
Here I am breaking with the order of the course a bit
But originally the course goes into model evaluation metrics and resampling
Personally I think this should come first, so I'm doing it
This script will take us through many model types:
- Linear (Logistic Regression, Linear Discriminate Analysis)
- Non Linear Algorithms (Decision Tree, Support Vector Machine, KNN)
- More advanced ensemble methods (Random Forest, Gradient Boosted Machine)

Since this is simply a spot check,
I will also be doing the bare minimum for model checking:
- Simple train test split
- Default model parameters
- Use accuracy from predicting test set

We will also be supplying a baseline model to compare these models to
The baseline will be simply predicting the majority class
In other cases, baseline models could be predicting the mean, median, etc.
'''

# Train test split
X = array[:, 0:8]
y = array[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                    test_size = 0.30, random_state = 42) 

# Make list for models
models = np.empty([10, 2], dtype = object)

# Linear models
models[0] = ['Logistic Regression', LogisticRegression(random_state = 1)]
models[1] = ['Linear Discriminant Analysis', LinearDiscriminantAnalysis()]
models[2] = ['SGD Classifier', SGDClassifier(max_iter = 5, 
tol = None, random_state = 1)]

# Nonlinear models
models[3] = ['Naive Bayes', MultinomialNB()]
models[4] = ['Decision Tree', DecisionTreeClassifier(max_features = 3, random_state = 1)]
models[5] = ['Support Vector Machine', LinearSVC(random_state = 1)]
models[6] = ['K Nearest Neighbors', KNeighborsClassifier()]

# More complex models	
models[7] = ['Neural Network', MLPClassifier(random_state = 1)]
models[8] = ['Random Forest', RandomForestClassifier(random_state = 1)]
models[9] = ['Gradient Boosted Machine', GradientBoostingClassifier(random_state = 1)]

# First print out baseline model: Predict majority class
counts = Series(y).value_counts()
# The code below gets the value of the majority class
# As well as it's frequency
majority, majority_count = counts.idxmax(), counts.max()
# Now make baseline model
baseline_model = majority_count / len(y)
print("\nBaseline Model Predicts Majority Class\
	with a probability of %0.2f\n" % baseline_model)

# Iterate through models, then fit & evaluate 
for name, model in models:

	# Fit model and make predictions
	fitted_model = model.fit(X_train, y_train)
	y_pred = fitted_model.predict(X_test)

	# Output predictive accuracy
	acc = fitted_model.score(X_test, y_test)
	print("\nAccuracy of %s model: %0.2f" % (name, acc))

'''
Since the baseline accuracy is 65%, 
only models who are noticably higher in accuracy should move on.
The candidates who have made the cut:
- Logistic Regression
- Linear Discriminant Analysis
- Random Forest
- Gradient Boosted Machine
'''