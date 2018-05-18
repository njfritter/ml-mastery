#!/usr/bin/env python3

###############################
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
#
# General style for python:
#
# snake_case for functions and variables, PascalCase for classes
###############################

# Welcome to my repo for the Mastering Machine Learning Python Mini Course
# Here I will be going through each part of the course
# So you can get a feel of the different parts


# Import packages before anything else
import pandas as pd
import numpy as np
from pandas import read_csv
from numpy import loadtxt
import urllib
import csv

# Define url and columns
url = 'https://goo.gl/bDdBiA'
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Read in data
data = read_csv(url, names = columns)

# Lesson 3: Understand Data w/ Descriptive Statistics
'''
This script have us load in our data and take a look at it
Some of the things we will be doing:
- Examining the first couple of rows and last couple of rows
- Check out the dimensions of the data 
- Look at different data types of the attributes
- Review distribution of data
- Calculate pairwise correlation between variables
'''

# Examine head and table of data
try:
    head = data.head(10)
    tail = data.tail(10)
    print("\nHere is the head of the data:\n", head)
    print("\nHere is the tail of the data:\n", tail)

except Exception as e:
    print("Error message:", e)

# Check dimensions of data
try:
    dim = data.shape
    print("\nThe shape of the data:\n" , dim)

except Exception as e:
    print("Error message:", e)

# Look at different data types of the attributes
try:
    dtypes = data.dtypes
    print("\nData type encountered: \n", dtypes) 

except Exception as e:
    print("Error message:", e)

# Review distribution of data
try:
    dist = data.describe()
    print("\nSummary statistics for each variable:\n", dist)

except Exception as e:
    print("Error message:", e)

# Calculate pairwise correlation
try:
    corr = data.corr()
    print("\nPairwise correlation between variables:\n", corr)

except Exception as e:
    print("Error message:", e)
