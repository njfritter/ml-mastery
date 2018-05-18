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
# Define url and columns
url = 'https://goo.gl/bDdBiA'
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])


# Lesson 2: Loading in Data through different methods
'''
This script will load in the data using three different methods:
1. pandas.read_csv()
2. numpy.loadtxt()
	a. Make sure to specify the delimiter, otherwise it throws an error
3. csv.reader()
	a.Original method can only read physical csvs
	b. We need to include the urllib.request.urlopen() method for the url
	c. THEN, to properly load the data, we need to create a generator object using codecs
	d. Then convert to a list, THEN to a DataFrame in pandas. Having fun yet?
'''

# Method 1
try:
    method = "pandas.read_csv()"
    print("\nLoading in data via the %s method using a url\n" % method)
    data = read_csv(url, names = columns)

    print("Data read successfully using the %s method" % method)

except Exception as e:
    print("Error message:", e)

# Method 2
try:
    method = "numpy.loadtxt()"
    print("\nLoading in data via the %s method using a url\n" % method)
    data = loadtxt(url, dtype = float)

    print("Data read successfully using the %s method" % method)

except Exception as e:
    print("Error message:", e)

# Method 3
try:
    method = "csv.reader()"
    print("\nLoading in data via the %s method using a url\n" % method)
    response = urllib.request.urlopen(url)
    data = csv.reader(codecs.iterdecode(response, 'utf-8'))
    data = pd.DataFrame(list(data), columns = columns)

    print("Data read successfully using the %s method" % method)