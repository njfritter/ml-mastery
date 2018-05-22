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
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns

# Define url and columns
url = 'https://goo.gl/bDdBiA'
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Read in data
data = read_csv(url, names = columns)

# Lesson 4: Understand Data w/ Visualization
'''
This script have us load in our data 
and visualize it in multiple different ways:
- Heatmap using the Seaborn package
- Scatterplot using the Seaborn package
- Scatterplot using the Pandas package
- Boxplot using the Seaborn package
- Boxplot using the builtin method for Pandas "read_csv"
- Histogram using the builtin method for Pandas "read_csv"
'''

# Method 1: Seaborn Heatmap
print("\n Now plotting a Heatmap using Seaborn\n")
try:
    # Plot correlation matrix on heatmap w/ custom diverging colormap
    f, ax = plt.subplots(figsize = (11, 9))
    c_map = sns.diverging_palette(100, 200, as_cmap = True)

    # Draw heatmap with mask and correct aspect ratio
    sns.heatmap(data, cmap = c_map, square = True,
    	xticklabels = True, yticklabels = True,
    	linewidths = 0.5, cbar_kws = {"shrink": 0.5}, ax = ax)
    #plt.title("Heatmap of Pre-Processed Variables: Seaborn Method", loc = 'center')
    ax.set_title("Heatmap of Pre-Processed Variables: Seaborn Method")

except Exception as e:
    print("Error message:", e)

# Method 2: Seaborn Scatterplot
print("\n Now plotting a Scatterplot using Seaborn\n")
try:	
    cols = data.columns.values
    sns.pairplot(data, 
        x_vars = cols,
    	y_vars = cols,
    	#hue = 'class',
    	palette = ('Red', 'Blue'),
    	markers = ["o", "D"])

    plt.title("Scatter Plot of Pre-Processed Variables: Seaborn Method", loc = 'center')

except Exception as e:
    print("Error message:", e)

# Method 3: Seaborn Boxplot
print("\n Now plotting a Boxplot using Seaborn\n")
try:
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

except Exception as e:
    print("Error message:", e)

# Method 4: Scatter Plot using Pandas
print("\n Now plotting a Scatterplot using Pandas\n")
try:
    graph = pd.plotting.scatter_matrix(data)
    plt.title("Scatter Plot of Pre-Processed Variables: Other Method", loc = 'center')

except Exception as e:
    print("Error message:", e)


# Method 5: Boxplot using builtin method
print("\n Now plotting a Boxplot using Pandas builtin method\n")
try:
    data.plot(kind = 'box')
    plt.title("Box Plot of Pre-Processed Variables: Other Method", loc = 'center')

except Exception as e:
    print("Error message:", e)

# Method 6: Histogram using builtin method
print("\n Now plotting a Histogram using Pandas builtin method\n")
try:    
    data.hist()
    plt.title("Histogram of Pre-Processed Variables: Other Method", loc = 'center')

    plt.show()
    plt.close()

except Exception as e:
    print("Error message:", e)