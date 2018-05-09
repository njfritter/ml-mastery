#!/usr/bin/env python3

'''
Mastering ML Python Mini Course
Inspired by the project here: 
https://s3.amazonaws.com/MLMastery/machine_learning_mastery_with_python_mini_course.pdf?__s=mxhvphowryg2sfmzus2q
By Nathan Fritter
Project will soon be found at: 
https://www.inertia7.com/projects/
General style for python:
snake_case for functions and variables, PascalCase for classes
'''

#
##
### Section: Regression
##
#


# Import packages
import numpy as np
import pandas as pd
import sys
import csv
import urllib
import codecs
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing

url = 'https://goo.gl/sXleFv'
columns = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])

dataframe = pd.read_csv(url, delim_whitespace = True, names = columns)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]

# Load in data using multiple methods
def head_tail_shape(data, method):
	try:
		# pandas.read_csv() & csv.reader() method
		print("\nHere is the head of the data for the %s method %s:\n" %(method), data.head(10))
		print("\nHere is the tail of the data for the %s method %s:\n" %(method), data.tail(10))
		print("\nHere is the shape of the data for Method %s:\n" %(method), data.shape)
	except AttributeError:
		# numpy.loadtxt() method
		print("\nHere is the head of the data for %s method :\n" %(method), data[0:9,])
		print("\nHere is the tail of the data for %s method :\n" %(method), data[0:9,])
		print("\nHere is the shape of the data for Method %s:\n" %(method), data.shape)

def read_data(src):

	'''
	This function will load in the data using three different methods:
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
	print("\nLoading in data via the %s method in pandas using a url\n" % (method))
	data = pd.read_csv(url, names = columns)

	head_tail(data, "pandas.read_csv()")

	# Method 2
	print("\nLoading in data via the loadtxt method in numpy using a url\n")
	data = np.loadtxt(url, dtype = float, delimiter = ',')

	head_tail(data, "numpy.loadtxt()")

	# Method 3
	print("\nLoading in data via the csv.reader() method using a url\n")
	response = urllib.request.urlopen(url)
	data = csv.reader(codecs.iterdecode(response, 'utf-8'))
	data = pd.DataFrame(list(data), columns = columns)

	head_tail(data, "csv.reader()")

def cross_validation(name, model, X, Y, scoring):
	# Automatically choosing 10-fold cross validation
	# May change that
	k_fold = model_selection.KFold(n_splits = 10, random_state = 1)	
	try:
		result = model_selection.cross_val_score(model, X, Y, cv = k_fold, scoring = scoring)
		if scoring == 'accuracy':
			print("\n%s of %s model:\n %.3f%% (+\-%.3f%%)" % 
				(scoring, name, result.mean() * 100.0, result.std() * 100.0))
		else:
			print("\n%s of %s model:\n %.3f (+\-%.3f)" % 
				(scoring, name, result.mean(), result.std()))
	except AttributeError:
		print("The %s model cannot perform cross validation with the %s metric" % (name, scoring))

# Define candidates for regression
def regression():
	# Now we will do a regression problem for one part
	# It deserves more, but this is all I can do for now
	# Most of the algorithms above can be used for regression
	# So I will use some of the ones I haven't used yet
	new_url = 'https://goo.gl/sXleFv'
	new_columns = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])

	dataframe = pd.read_csv(new_url, delim_whitespace = True, names = new_columns)
	array = dataframe.values
	X = array[:, 0:13]
	Y = array[:, 13]

	k_fold = model_selection.KFold(n_splits = 10, random_state = 7)
	models = np.empty([6, 2], dtype = 'object')
	models[0] = ['K Nearest Neighbors', neighbors.KNeighborsRegressor()]
	models[1] = ['Linear Regression', linear_model.LinearRegression()]
	models[2] = ['Ridge Regression', linear_model.Ridge()]
	models[3] = ['Support Vector Regressor', svm.LinearSVR()]
	models[4] = ['Random Forest Regressor', ensemble.RandomForestRegressor()]
	models[5] = ['Gradient Boosted Trees', ensemble.GradientBoostingRegressor()]

	for name, model in models:
		# Different model metrics
		for scoring in ('neg_mean_squared_error', 'explained_variance'):
			cross_validation(name, model, X, Y, scoring)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'test':
        	stand_norm_test()
        elif sys.argv[1] == 'sig':
        	significant_variables()
        elif sys.argv[1] == 'reg':
        	regression()
        else:
        	print("Incorrect keyword; please enter a valid keyword.")
    else:
    	print("Incorrect number of keywords; please enter one keyword after the program name.")


