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
### SECTION: Exploratory Analysis
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

# Define url and columns
url = "https://goo.gl/vhm1eU"
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Define data and columns
diabetes_data = 'data/diabetes.csv'
diabetes_columns = np.array(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
	'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])

# Read in data using url
#data = pd.read_csv(url, names = columns)

# Read in using csv file
data = pd.read_csv(diabetes_data, names = diabetes_columns)
array = data.values



# Describe the data
def describe_data(data):
	'''
	This function will output:
	- Datatype
	- Shape
	- Summary statistics (?)
	'''
	try:
		print("\nData type encountered: %s\n" % (type(data)))
		print("\nThe shape of my data: %s\n" % (data.shape))
		print("Data below:\n", data)
		description = data.describe()
	
		print("\nSummary statistics for each variable:\n", description)
		print("\nData types of the variables:\n", data_types)

	except TypeError:
		print("Incompatible data type")	

	try:
		data_types = data.dtypes
		correlation = data.drop(['Outcome'], axis = 1).corr(method = 'pearson')
		print("\nPearson correlation between pairs of variables:\n", correlation)

	except TypeError:
		print("Incompatible data type")
	

#
##
### PART TWO: More in depth visual analysis (using Seaborn and other methods)
##
#


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

	# Correlation (first calculate correlation)
	correlation = data.corr(method = 'pearson')
	plot_data_seaborn(correlation, "Heatmap")#

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



#
##
### PART THREE: Preparing Data for Modeling via Pre-Processing
##
#

'''
The unprocessed data may not be in the best shape for modeling
Sometimes the best interpretation of the inherent structure of the data come via transformations 
Scikit learn provides two standards for transforming data (useful in different cases)
They are "Fit and Multiple Transform" and "Combined Fit-and-Transform"
Will go into more detail later

Methods for preparing data for modeling
1. Standardize numerical data (i.e. mean of 0 standard deviation of 1) using scale & center options
2. Normalize numerical data (i.e. to range of 0 to 1) using range option
3. Explore advanced feature engineering such as Binarizing
'''

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
        if sys.argv[1] == 'describe':
            describe_data(data)
        elif sys.argv[1] == 'visualize':
            visualize_data()
        elif sys.argv[1] == 'transform':
            transform_data()
        else:
        	print("Incorrect keyword; please enter a valid keyword.")
    else:
    	print("Incorrect number of keywords; please enter one keyword after the program name.")

