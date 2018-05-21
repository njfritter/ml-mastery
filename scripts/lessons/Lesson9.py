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

import numpy as np
import pandas as pd
from pandas import read_csv, Series
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
# Define url and columns
url = 'https://goo.gl/bDdBiA'
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Read in data
data = read_csv(url, names = columns)
array = data.values

####################################
# Lesson 9: More in Depth Model Metrics for Comparison
####################################

'''
Originally this part of the course was its own part:
to use various model metrics and compare models to each other
However I think comparison is intuitive enough so I've simply
been doing it throughout the model evaluation process.

So I will do some more in depth methods for comparison
to see if any other of the models can be eliminated.
'''

# Divide data into attributes and predictor
X = array[:, 0:8]
y = array[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

# Make list for models
models = np.empty([4, 2], dtype = object)

# Define models
models[0] = ['Logistic Regression', LogisticRegression(random_state = 1)]
models[1] = ['Linear Discriminant Analysis', LinearDiscriminantAnalysis()]	
models[2] = ['Random Forest', RandomForestClassifier(random_state = 1)]
models[3] = ['Gradient Boosted Machine', GradientBoostingClassifier(random_state = 1)]


# Iterate through models, then fit & evaluate 
for name, model in models:

    # Fit model and make predictions for next section
    fitted_model = model.fit(X_train, y_train)
    y_pred = fitted_model.predict(X_test)
    # Classification reports & confusion matrix here
    try:
        class_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nClassification report for %s:\n" % name, class_report)
        print("\nConfusion matrix for %s:\n" % name, conf_matrix)
    except AttributeError:
        print("The %s model is not compatible with these functions")        

    # ROC Curves
    try:
        fpr, tpr, threshold = roc_curve(y_true = y_test, y_score = y_pred, pos_label = 1)
        roc_auc = auc(fpr, tpr)
        
        plt.title('ROC Curve: %s' % name)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0,1], [0,1], 'r--') # Add diagonal line
        plt.plot([0,0], [1,0], 'k--', color = 'black')
        plt.plot([1,0], [1,1], 'k--', color = 'black')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
    except:
        print("The %s model is not compatible with the %s method" % (name, roc_curve.__name__))

    try:
        importances = fitted_model.feature_importances_
        #std = np.std([tree.feature_importances_ for tree in fitted_model.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]
        print("\nFeature importances for %s model:" %(name))
        for f in range(X.shape[1]):
            print("%d. Feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        plt.figure()
        plt.title("Feature Importances for %s" % name)
        plt.bar(range(X.shape[1]), importances[indices], color = 'r', align = 'center')
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.show()
    except AttributeError: 
        print("The %s model has no attribute: %s" % (name, "feature_importances_"))

'''
Now we have more in depth metrics to make comparisons and inferences
For example:
- The confusion matrix 

'''