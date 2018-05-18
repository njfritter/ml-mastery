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
import numpy as np
import pandas as pd

# Lesson 1: Numpy and Pandas Practice

def basic_output(data):
	'''
	This function will output:
	- Datatype
	- Shape
	- Summary statistics (?)
	'''
	try:
		print("\nData type encountered: %s\n" % (type(data)))
		print("\nThe shape of my %s: %s\n" % (type(data) , data.shape))
		print("Data below:\n", data)

	except TypeError:
		print("Incompatible data type")	

print("########################################")
print("This is Part 1: Getting Around in Python")
print("########################################")

my_array = np.array([[1, 2, 3], [4, 5, 6]])
row_names = np.array(['a', 'b'])
col_names = np.array(['one', 'two', 'three'])
my_df = pd.DataFrame(my_array, index = row_names, columns = col_names)
basic_output(my_df)

my_series = np.array([7, 8, 9])
series_index = np.array(['bears', 'beets', 'battlestar galatica'])
my_pd_series = pd.Series(my_series, index = series_index)
basic_output(my_pd_series)
