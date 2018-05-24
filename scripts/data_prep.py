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
###############################

# Import packages
import numpy as np
import pandas as pd
from urllib.request import urlopen
from urllib.error import HTTPError
from sklearn.preprocessing import normalize, StandardScaler, scale, MinMaxScaler
import helper_functions as hf

# The mini course used a url, however it got taken down and replaced
# So we'll try the link first, then default to a physical CSV if the url fails
try:
	diabetes_url = 'https://goo.gl/bDdBiA'
	diabetes_columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

	# Read in data using url
	diabetes_data = pd.read_csv(urlopen(diabetes_url), names = diabetes_columns)
	print("\nReading from URL successful\n")

except HTTPError:
	print("\nThe URL link doesn't exist, reading in physical CSV\n")
	
	# Define data and columns
	diabetes_file = 'data/diabetes.csv'
	diabetes_columns = np.array(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
		'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])

	# Read in using csv file
	diabetes_data = pd.read_csv(diabetes_file, names = diabetes_columns)
	print("Physical CSV read")

# Coerce errors
for column in diabetes_data:
	pd.to_numeric(column, errors = 'coerce')
"""
# Separate into input and output components
array = data.values
data_features = array[:, 0:8]
data_class = array[:, 8]	
"""
# Separate into input and output components
data_features = diabetes_data.iloc[:, diabetes_data.columns != 'class']
data_class = diabetes_data.iloc[:, diabetes_data.columns == 'class']

# Split data into training and testing set
train_features, test_features, train_class, test_class = hf.split_train_test(data_features,
																			data_class,
																			test_size = 0.30,
																			random_state = 42)


# Clean test sets to avoid future warning labels
train_class = train_class.values.ravel()
test_class = test_class.values.ravel()

# Normalized dataframe
data_norm = normalize(diabetes_data)
train_features_norm = normalize(train_features)
test_features_norm = normalize(test_features)

# Scaled data; will standardize data without any parameter specifications
# Will output same thing if "with_mean = True" and "with_std = True" specified
data_scaled = scale(diabetes_data)
train_features_scaled = scale(train_features)
test_features_scaled = scale(test_features)

# Standardized data; creates a standardizer and fits to the covariates
standard_scaler = StandardScaler()
standard_scaler.fit(diabetes_data)
data_stand = standard_scaler.transform(diabetes_data)

train_features_stand = standard_scaler.fit_transform(train_features)
test_features_stand = standard_scaler.fit_transform(test_features)

# MinMax Scaling
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(diabetes_data)
data_minmax = minmax_scaler.transform(diabetes_data)

train_features_minmax = minmax_scaler.fit_transform(train_features)
test_features_minmax = minmax_scaler.fit_transform(test_features)


