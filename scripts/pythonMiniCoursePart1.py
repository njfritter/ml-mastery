# ML Mastery Python MiniCourse
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
import csv
import urllib
import codecs
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing

# Define url and columns
url = "https://goo.gl/vhm1eU"
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Read in data
data = pd.read_csv(url, names = columns)
array = data.values

def basic_output(data):
	try:
		print("\nData type encountered: %s\n" % (type(data)))
		print("\nThe shape of my %s: %s\n" % (type(data) , data.shape))
		print("Data below:\n", data)

	except TypeError:
		print("Incompatible data type")	
def read_data(url, method):

	# Method one: Pandas read_csv() function
	if method == 1:
		print("\nLoading in data via the read_csv method in pandas using a url\n")
		data = pd.read_csv(url, names = columns)

	# Method two: use Numpy's loadtxt() method
	# Make sure to specify the delimiter, otherwise it throws an error
	if method == 2:
		print("\nLoading in data via the loadtxt method in numpy using a url\n")
		data = np.loadtxt(url, dtype = float, delimiter = ',')

	# Method three: Load in using csv.reader() function
	# Since the original method can only read physical csvs
	# We need to include the urllib.request.urlopen() method for the url
	# THEN, to properly load the data, we need to create a generator object using codecs
	# Then convert to a list, THEN to a DataFrame in pandas. Having fun yet?
	if method == 3:
		print("\nLoading in data via the csv.reader() method using a url\n")
		response = urllib.request.urlopen(url)
		data = csv.reader(codecs.iterdecode(response, 'utf-8'))
		data = pd.DataFrame(list(data), columns = columns)

	print("\nHere is the shape of the data for Method %s:\n" %(method), data.shape)

	# For methods 1 and 3
	try:
		print("\nHere is the head of the data for Method %s:\n" %(method), data.head(10))
		print("\nHere is the tail of the data for Method %s:\n" %(method), data.tail(10))

	# For method 2
	except AttributeError:	
		print("\nHere is the head of the data for Method %s:\n" %(method), data[0:9,])
		print("\nHere is the tail of the data for Method %s:\n" %(method), data[0:9,])

def plot_data_seaborn(data, method):

	try:
		if method == "Heatmap":
			# Plot correlation matrix on heatmap w/ custom diverging colormap
			f, ax = plt.subplots(figsize = (11, 9))
			c_map = sns.diverging_palette(100, 200, as_cmap = True)

			# Draw heatmap with mask and correct aspect ratio
			sns.heatmap(data, cmap = c_map, square = True,
				xticklabels = True, yticklabels = True,
				linewidths = 0.5, cbar_kws = {"shrink": 0.5}, ax = ax)
			#plt.title("Heatmap of Pre-Processed Variables: Seaborn Method", loc = 'center')
			ax.set_title("Heatmap of Pre-Processed Variables: Seaborn Method")

		if method == "Scatterplot":
			cols = data.columns.values
			sns.pairplot(data, 
				x_vars = cols,
				y_vars = cols,
				hue = 'class',
				palette = ('Red', 'Blue'),
				markers = ["o", "D"])
			plt.title("Scatter Plot of Pre-Processed Variables: Seaborn Method", loc = 'center')

		if method == "Boxplot":
			f , ax = plt.subplots(figsize = (11, 20))

			ax.set_facecolor('#fafafa')
			ax.set(xlim = (-0.05, 50))
			#plt.ylabel('Dependent Variables')
			ax.set_ylabel('Dependent Variables')
			ax.set_title("Box Plot of Pre-Processed Variables: Seaborn Method")
			#plt.title("Box Plot of Pre-Processed Variables: Seaborn Method", loc = 'center')
			ax = sns.boxplot(data = data, orient = 'h', palette = 'Set2')

		plt.show()
		plt.close()

	except ValueError:
		print("Invalid method entered")

def plot_data_other(data, method):

	try:
		if method == "Scatterplot":
			graph = pd.plotting.scatter_matrix(data)
			plt.title("Scatter Plot of Pre-Processed Variables: Other Method", loc = 'center')

		if method == "Boxplot":
			# Box and whisker plot
			data.plot(kind = 'box')
			plt.title("Box Plot of Pre-Processed Variables: Other Method", loc = 'center')

		if method == "Histogram":
			data.hist()
			plt.title("Histogram of Pre-Processed Variables: Other Method", loc = 'center')

		plt.show()
		plt.close()

	except ValueError:
		print("Invalid method entered")

def transform(data, method):

	try:
		if method == "Standardized":
			# Create a standardizer and fit to the covariates
			scaler = preprocessing.StandardScaler().fit(data)
			transformed_data = scaler.transform(data)

		if method == "Normalized":
			# Normalize data and summarize
			transformed_data = preprocessing.normalize(data)

		if method == "Scaled":
			# Scale data and summarize
			# This will standardize data without any parameter specifications
			# Will also do this if "with_mean = True" and "with_std = True" specified
			transformed_data = preprocessing.scale(data)

		np.set_printoptions(precision = 3)
		print("\n%s Output:\n" % method, transformed_data[0:5, :], "\n")

	except ValueError:
		print("Invalid method entered")

def the_basics():
	my_array = np.array([[1, 2, 3], [4, 5, 6]])
	row_names = np.array(['a', 'b'])
	col_names = np.array(['one', 'two', 'three'])
	my_df = pd.DataFrame(my_array, index = row_names, columns = col_names)
	basic_output(my_df)

	my_series = np.array([7, 8, 9])
	series_index = np.array(['bears', 'beets', 'battlestar galatica'])
	my_pd_series = pd.Series(my_series, index = series_index)
	basic_output(my_pd_series)

#
##
###
#### PART TWO: Loading in data from CSV (Pima Indians onset of diabetes dataset, UCI ML repo)
###
##
#

def load_data():

	# Method 1
	read_data(url, 1)

	# Method 2
	read_data(url, 2)

	# Method 3
	read_data(url, 3)

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

def explore_data():

	# Remove class: Not necessary or meaningful here
	# data = data.iloc[:, 0:8]
	description = data.describe()
	data_types = data.dtypes
	correlation = data.corr(method = 'pearson')
	
	print("\nSummary statistics for each variable:\n", description)
	print("\nData types of the variables:\n", data_types)

	"""
	# One way to show distribution of data
	# Not great, better method shown in next part
	# Originally had this part shown
	# But the next part is better
	for column in data.select_dtypes(include = int): 
		freq = data[column].value_counts().sort_index()
		print("\nFrequency of the column {} with graph:\n".format(column), freq)
		freq.plot(kind = 'bar')
		plt.show()
	"""
	plot_data_seaborn(correlation, "Heatmap")

#
##
###
#### PART FOUR: More in depth visual analysis
###
##
#

def visualize_data():

	# Here we will do a bit more in depth visualization stuffs	
	# Scatterplot matrix through pandas
	plot_data_other(data, "Scatterplot")

	# Scatterplot matrix through Seaborn
	plot_data_seaborn(data, "Scatterplot")

	# Box and whisker plot (built in method)
	plot_data_other(data, "Boxplot")

	# Seaborn box and whisker plot
	plot_data_seaborn(data, "Boxplot")

	# Histogram (better version of frequency counts; shows all at once)
	plot_data_other(data, "Histogram")

#
##
###
#### PART FIVE: Preparing Data for Modeling via Pre-Processing
###
##
#

# The unprocessed data may not be in the best shape for modeling
# Sometimes the best interpretation of the inherent structure of the data come via transformations 
# Scikit learn provides two standards for transforming data (useful in different cases)
# They are "Fit and Multiple Transform" and "Combined Fit-and-Transform"
# Will go into more detail later

"""
Methods for preparing data for modeling
1. Standardize numerical data (i.e. mean of 0 standard deviation of 1) using scale & center options
2. Normalize numerical data (i.e. to range of 0 to 1) using range option
3. Explore advanced feature engineering such as Binarizing
"""

def transform_data():

	# Separate into input and output components
	X = array[:, 0:8]
	Y = array[:, 8]	

	# Standardize
	transform(X, "Standardized")

	# Normalize
	transform(X, "Normalized")

	# Scale
	transform(X, "Scaled")


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'basic':
            the_basics()
        elif sys.argv[1] == 'load':
            load_data()
        elif sys.argv[1] == 'explore':
            explore_data()
        elif sys.argv[1] == 'visualize':
            visualize_data()
        elif sys.argv[1] == 'transform':
            transform_data()

