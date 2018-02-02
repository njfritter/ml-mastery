# ML Mastery Python MiniCourse!! Woo
# General style for python:
# snake_case for functions and variables, PascalCase for classes

#
##
###
#### PART ONE: Numpy and Pandas Practice
###
##
#

import numpy as np
import pandas as pd
import sys

def part_one():
	my_array = np.array([[1, 2, 3], [4, 5, 6]])
	row_names = ['a', 'b']
	col_names = ['one', 'two', 'three']
	my_df = pd.DataFrame(my_array, index = row_names, columns = col_names)
	print("\nMy DataFrame:\n", my_df)
	print("\nThe shape of my DataFrame:\n", my_df.shape)

	my_series = np.array([7, 8, 9])
	series_index = ['bears', 'beets', 'battlestar galatica']
	my_pd_series = pd.Series(my_series, index = series_index)
	print("\nMy Series:\n", my_pd_series)
	print("\nThe shape of my Series:\n", my_pd_series.shape)

#
##
###
#### PART TWO: Loading in data from CSV (Pima Indians onset of diabetes dataset, UCI ML repo)
###
##
#

# Define url and columns
url = "https://goo.gl/vhm1eU"
columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
import csv
import urllib
import codecs

def part_two():
	datas = []
	# Method one: Pandas read_csv() function
	print("\nLoading in data via the read_csv method in pandas using a url\n")
	data = pd.read_csv(url, names = columns)
	datas.append(('Method 1',data))

	# Method two: use Numpy's loadtxt() method
	# Make sure to specify the delimiter, otherwise it throws an error
	print("\nLoading in data via the loadtxt method in numpy using a url\n")
	data = np.loadtxt(url, dtype = float, delimiter = ',')
	datas.append(('Method 2', data))

	# Method three: Load in using csv.reader() function
	# Since the original method can only read it physical csvs
	# We need to include the urllib.request.urlopen() method for the url
	# THEN, to properly load the data, we need to create a generator object using codecs
	# Then convert to a list, THEN to a DataFrame in pandas
	# Having fun yet?
	print("\nLoading in data via the csv.reader() method using a url\n")
	response = urllib.request.urlopen(url)
	data = csv.reader(codecs.iterdecode(response, 'utf-8'))
	new_data = pd.DataFrame(list(data), columns = columns)
	datas.append(('Method 3', new_data))


	for method, data in datas:
		# For methods 1 and 3
		try:
			print("\nHere is the shape of the data for %s:\n" %(method), data.shape)
			print("\nHere is the head of the data for %s:\n" %(method), data.head(10))
			print("\nHere is the tail of the data for %s:\n" %(method), data.tail(10))

		# For method 2
		except AttributeError:	
			print("\nHere is the shape of the data for %s:\n" %(method), data.shape)
			print("\nHere is the head of the data for %s:\n" %(method), data[0:9,])
			print("\nHere is the tail of the data for %s:\n" %(method), data[0:9,])


#
##
###
#### PART THREE: Understand Data w/ Descriptive Statistics
###
##
#

# Now we must look further into the data to see what it is telling us
# The shape and head/tail were already printed out in Part 2
# Now lets go more in depth
# And also add graphs!

import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
def part_three():

	data = pd.read_csv(url, names = columns)
	description = data.describe()
	data_types = data.dtypes
	correlation = data.corr()
	
	print("\nSummary statistics for each variable:\n", description)
	print("\nData types of the variables:\n", data_types)
	print("\nPairwise correlation between variables:\n", correlation)
	
	# One way to show distribution of data
	# Not great, better method shown in next part
	# Originally had this part shown
	# Commenting out because while it does the job
	# The next part is better
	"""
	for column in data.select_dtypes(include = int): 
		freq = data[column].value_counts().sort_index()
		print("\nFrequency of the column {} with graph:\n".format(column), freq)
		freq.plot(kind = 'bar')
		plt.show()
	"""

#
##
###
#### PART FOUR: More in depth visual analysis
###
##
#

def part_four():

	# Here we will do a bit more in depth visualization stuffs
	data = pd.read_csv(url, names = columns)
	
	# Scatterplot matrix
	graph = pd.plotting.scatter_matrix(data)
	plt.show()

	# Box and whisker plot
	data.plot(kind = 'box')
	plt.show()

	# Histogram (better version of frequency counts; shows all at once)
	data.hist()
	plt.show()


#
##
###
#### PART FIVE: Preparing Data for Modeling via Pre-Processing
###
##
#

# One's data may not be in the best shape for modeling
# Sometimes transformations are needed 
# In order to best present the inherent structure of the data to the model
# Scikit learn provides two standards for transforming data (useful in different cases)
# They are "Fit and Multiple Transform" and "Combined Fit-and-Transform"
# Will go into more detail later

"""
Methods for preparing data for modeling
1. Standardize numerical data (i.e. mean of 0 standard deviation of 1) using scale & center options
2. Normalize numerical data (i.e. to range of 0 to 1) using range option
3. Explore advanced feature engineering such as Binarizing
"""

from sklearn import preprocessing
def part_five():
	# Here we will standardize, normalize & scale the data the data
	data = pd.read_csv(url, names = columns)
	array = data.values

	# Separate into input and output components
	X = array[:, 0:8]
	Y = array[:, 8]	

	# Create a standardizer and fit to the covariates
	scaler = preprocessing.StandardScaler().fit(X)
	standard_X = scaler.transform(X)

	# Summarize transformed data
	np.set_printoptions(precision = 3)
	print("\nStandarized Output:\n", standard_X[0:5, :], "\n")

	# Normalize data and summarize
	norm_X = preprocessing.normalize(X)
	print("\nNormalized Output:\n", norm_X[0:5, :], "\n")

	# Scale data and summarize
	# This will standardize data without any parameter specifications
	# Will also do this if "with_mean = True" and "with_std = True" specified
	scale_X = preprocessing.scale(X)
	print("\nScaled Output:\n", scale_X[0:5, :], "\n")	



if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == '1':
            part_one()
        elif sys.argv[1] == '2':
            part_two()
        elif sys.argv[1] == '3':
            part_three()
        elif sys.argv[1] == '4':
            part_four()
        elif sys.argv[1] == '5':
            part_five()

