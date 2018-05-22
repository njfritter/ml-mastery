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
import pickle
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  train_test_split

# Define url and columns
url = 'https://goo.gl/bDdBiA'
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Read in data
data = read_csv(url, names = columns)
array = data.values

# Divide data into attributes and predictor
X = array[:, 0:8]
y = array[:, 8]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

####################################
# Lesson 12: Finalize and Save Model
####################################

'''
Last but not least, we will save our final chosen model to be able to be used again
Since the final set of models didn't have much separation, 
I am going to go with the simplest model, Logistic Regression
Because it is the easiest to explain and train
Only one parameter needs to be specified from the parameter tuning: C = 1.25
'''

final_model = LogisticRegression(random_state = 1, C = 1.25)
final_model.fit(X_train, y_train)

# Save model to disk
filename = './models/finalized_Logistic_Regression_Model.sav'
pickle.dump(final_model, open(filename, 'wb'))

# Wait for it...
# Wait for it...
# Wait for it...

# Load model back in!
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print("Test set score:", result)