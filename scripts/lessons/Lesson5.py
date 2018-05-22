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


# Import packages before anything else
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn import preprocessing

# Define url and columns
url = 'https://goo.gl/bDdBiA'
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Read in data
data = read_csv(url, names = columns)

########################################################
# Lesson 5: Prepare Data for Modeling by Pre-Processing
########################################################
'''
This script have us load in our data 
and utitlize different transformation methods.
Data isn't always in an actionable format starting out, 
so sometimes transforming data becomes necessary.
The methods we will look at here:
- Standardize data
- Normalize data
- More advanced feature engineering such as Binarizing
'''

# Method 1: Standardizing
try:
    # Create a standardizer and fit to the covariates
    method = "Standardized"
    scaler = preprocessing.StandardScaler().fit(data)
    transformed_data = scaler.transform(data)
    
    np.set_printoptions(precision = 3)
    print("\n%s Output:\n" % method, transformed_data[0:5, :], "\n")

except Exception as e:
    print("Error message:", e)

# Method #2: Normalizing
try:
    method = "Normalized"
    transformed_data = preprocessing.normalize(data)
    
    print("\n%s Output:\n" % method, transformed_data[0:5, :], "\n")

except Exception as e:
    print("Error message:", e)

# Method #3: Scaling
# This will standardize data without any parameter specifications
# Will also do this if "with_mean = True" and "with_std = True" specified

try:
    method = "Scaled"
    transformed_data = preprocessing.scale(data)

    print("\n%s Output:\n" % method, transformed_data[0:5, :], "\n")

except Exception as e:
    print("Error message:", e)
