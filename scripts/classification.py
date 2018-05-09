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
### SECTION: Classification
##
#

# The data used to train a model should not be then predicted using the model
# Because the point is to make predictions on unseen data to see how well it generalizes
# To do this we use resampling methods to split the data into training & testing sets
# Then fit a model and evaluate its performance on the testing set

"""
Methods for Re-sampling data
1. Split data once into training and testing sets
2. Use k-fold cross-validation to create k different train test splits to train k different models
3. Use leave one out cross-validation: 
	a. Every data point is held out once with the rest of the data used to fit a model 
	b. Thus n models created, with each trained on n - 1 data points
	c. Better for smaller datasets

Methods for Evaluating Algorithm Metrics
1. Accuracy and LogLoss metrics for classdification
2. Generation of confusion matric and classification report
3. Root Mean Square Error (RMSE) and R squared matrics for regression
"""

import numpy as np
import pandas as pd
import sys
from sklearn import (model_selection, linear_model, metrics, discriminant_analysis,
neural_network, tree, svm, naive_bayes, ensemble)
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import pickle

# Define url and columns
url = 'https://goo.gl/bDdBiA'
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

# Read in data and split up so we don't do this over and over
data = pd.read_csv(url, names = columns)
array = data.values

# Separate into input and output components
X = array[:, 0:8]
Y = array[:, 8]	

# Train test split for evaluation metrics
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
X, Y, test_size = 0.33, random_state = 42)

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
	"""
	else:
		except Exception as e:
			print(e)
	"""
def classification_report(name, labels, predictions):
	try:
		class_report = metrics.classification_report(labels, predictions)
		print("\nClassification Report for %s model:\n" % (name), class_report)
	except:
		print("The %s model is not compatible with the %s method" % (name, metrics.classification_report.__name__))
def confusion_matrix(name, labels, predictions):
	try:
		conf_matrix = metrics.confusion_matrix(labels, predictions)
		print("\nConfusion Matrix for %s model:\n" % (name), conf_matrix)
	except:
		print("The %s model is not compatible with the %s method" % (name, metrics.confusion_matrix.__name__))

def receiver_operating_characteristic(name, labels, predictions):
	# ROC Curves
	try:
		fpr, tpr, threshold = metrics.roc_curve(y_true = labels, y_score = predictions, pos_label = 1)
		roc_auc = metrics.auc(fpr, tpr)
		
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
		print("The %s model is not compatible with the %s method" % (name, metrics.roc_curve.__name__))

def feature_importances(name, trained_model, X, Y):
	try:
		importances = trained_model.feature_importances_
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

def save_model(name, trained_model):
	# Here we will use the "pickle" function to save the model
	try:
		pickle.dump(trained_model, open('final_%s_model.sav' % (name), 'wb'))
	except:
		print("The %s model is not compatible with the %s method" % (name, pickle.dump.__name__))

def load_model(name):
	# Now we will load the model and return it
	try:
		loaded_model = pickle.load(open('final_%s_model.sav' % (name), 'rb'))
		return loaded_model
	except:
		print("The %s model is not compatible with the %s method" % (name, pickle.load.__name__))

def model_score(name, trained_model, test_data, labels):
	# Score model
	try:
		result = trained_model.score(test_data, labels)
		print("Score for %s model:\n" % name, result)
	except:
		print("The %s model is not compatible with the %s method" % (name, score.__name__))

def spot_check():
	# Here we will fit a Logistic Regression model using 10 fold cross validation
	# As well as a Linear Discriminant Analysis model & compare
	models = np.empty([8, 2], dtype = object)

	# Simpler models
	models[0] = ['Logistic Regression', linear_model.LogisticRegression(random_state = 1)]
	models[1] = ['Linear Discriminant Analysis', discriminant_analysis.LinearDiscriminantAnalysis()]
	models[2] = ['Naive Bayes', naive_bayes.MultinomialNB()]
	models[3] = ['Decision Tree', tree.DecisionTreeClassifier(max_features = 3, random_state = 1)]

	# More complex models	
	models[4] = ['Neural Network', neural_network.MLPClassifier(random_state = 1)]
	models[5] = ['Ridge Classifier', linear_model.RidgeClassifier(random_state = 1)]
	models[6] = ['SGD Classifier', linear_model.SGDClassifier(max_iter = 5, 
		tol = None, random_state = 1)]
	models[7] = ['Support Vector Machine', svm.LinearSVC(random_state = 1)]
	
	# Fit & evaluate models
	for name, model in models:
		# Different model metrics
		for scoring in ('accuracy', 'roc_auc'):
			cross_validation(name, model, X, Y, scoring)

		# Fit model and make predictions
		fitted_model = model.fit(X_train, Y_train)
		Y_pred = fitted_model.predict(X_test)
		
		# Classification report & Confusion Matrix
		classification_report(name, Y_test, Y_pred)
		confusion_matrix(name, Y_test, Y_pred)

	""" 
	Best performing models:
	1. Logistic Regression
	2. Linear Discriminant Analysis
	3. Ridge Classifier
	"""

#
##
### PART SEVEN: Improve Accuracy with Ensemble Predictions
##
#

# The previous part trained single models for evaluation
# This is usually sufficient, but one can also combine predictions from multiple equivalent models
# Some models are built in with this capacity, e.g:
# Random Forest for bagging, Stochastic Gradient Boosting for Boosting
# Another type of ensemble is called voting
# Where the predictions from multiple different models are combined (Done in next part)

def ensemble_models():
	# Here let's implement some ensemble methods to potentially improve accuracy
	# And get a better idea of the inherent structure of the data

	models = np.empty([4, 2], dtype = object)
	# Boosting ensembles
	models[0] = ['Gradient Boosted Machine', ensemble.GradientBoostingClassifier(random_state = 1)]
	models[1] = ['AdaBoost Classifier', ensemble.AdaBoostClassifier(random_state = 1)]

	# Bagging Ensembles
	# Even though the decision tree didn't do so well, a random forest might
	n_trees = 100
	models[2] = ['Random Forest', ensemble.RandomForestClassifier(n_estimators = n_trees,	max_features = 3, random_state = 1)]
	models[3] = ['Extra Trees Classifier', ensemble.ExtraTreesClassifier(n_estimators = n_trees, max_features = 3, random_state = 1)]

	# Fit & evaluate models
	for name, model in models:
		# Different model metrics
		for scoring in ('accuracy', 'roc_auc'):
			cross_validation(name, model, X, Y, scoring)

		# Fit model and make predictions
		fitted_model = model.fit(X_train, Y_train)
		Y_pred = fitted_model.predict(X_test)
		
		# Classification report & Confusion Matrix
		classification_report(name, Y_test, Y_pred)
		confusion_matrix(name, Y_test, Y_pred)		

	"""
	All of them achieve pretty much the same results as the simpler models
	Yet Logistic Regression still does slightly better than all of them
	I will do a Voting Classifier to see if I can combine any results here to get better results
	Otherwise logistic regression is the way to go
	"""

#
##
### PART EIGHT: Voting Classifier: Can we combine multiple models to achieve better performance?
##
#

# Technically this is an emsemble method, but I wanted to include this in a separate part
# In order to see the results of the initial model fitting
# And gather the models that did well
# Model diversity is key here
# Because the point of combining models is to reduce generalization error
# And similar models will not achieve that
# There are different voting methods as well (hard vs soft)
# All will be attempted

def voting_ensemble():
	# Last but not least, let's combine some of these models 
	# To try for better predictive performance
	n_trees = 100
	models = np.empty([2, 2], dtype = 'object')

	# Voting ensembles
	# Number 1: Hard Vote (Predicted class labels used for majority rule voting)
	models[0] = ['Voting Classifier 1', ensemble.VotingClassifier(estimators = [
		('lr', linear_model.LogisticRegression(random_state = 1)),
		('gbm', ensemble.GradientBoostingClassifier(random_state = 1)),
		], voting = 'hard')]

	# Number 2: Soft Vote (Argmax of sums of predicted probabilities used)
	# Recommended for ensemble of well-calibrated classifiers
	models[1] = ['Voting Classifier 2', ensemble.VotingClassifier(estimators = [
		('lda', discriminant_analysis.LinearDiscriminantAnalysis()),
		('rf', ensemble.RandomForestClassifier(random_state = 1, n_estimators = n_trees, max_features = 3))
		], voting = 'soft')]

	# Number 3: Soft Vote with weights
	# Some models will be more valuable than others

	# Fit & evaluate models
	for name, model in models:
		# Different model metrics
		for scoring in ('accuracy', 'roc_auc'):
			cross_validation(name, model, X, Y, scoring)

		# Fit model and make predictions
		fitted_model = model.fit(X_train, Y_train)
		Y_pred = fitted_model.predict(X_test)
		
		# Classification report & Confusion Matrix (needs separate training and evaluation process)
		classification_report(name, Y_test, Y_pred)
		confusion_matrix(name, Y_test, Y_pred)

#
##
### PART NINE: Pick a Final Model and Save
##
#

# Now that we have the results from many different model evalutions and testing, 
# Let's pick 1-2 final ones to save
# I will do one linear and one non-linear model 
# Because the training time difference is negligible
# Let's do Logistic Regression and a Random Forest
# Here we will do extra stuff since this is the final model chose
def final_models():

	n_trees = 100
	models = np.empty([2, 2], dtype = 'object')
	models[0] = ['Logistic_Regression', linear_model.LogisticRegression(random_state = 1)]
	models[1] = ['Random_Forest', ensemble.RandomForestClassifier(random_state = 1, n_estimators = n_trees, max_features = 3)]

	# Train models and save to disk
	for name, model in models:
		model.fit(X_train, Y_train)
		save_model(name, model)

	# A while later...
	for name, model in models:
		loaded_model = load_model(name)
		# Model score
		model_score(name, loaded_model, X_test, Y_test)

		# Classification report & Confusion Matrix (needs separate training and evaluation process)
		Y_pred = loaded_model.predict(X_test)
		classification_report(name, Y_test, Y_pred)
		confusion_matrix(name, Y_test, Y_pred)

		# Generate ROC Curves
		receiver_operating_characteristic(name, Y_test, Y_pred)

		# Feature importances
		feature_importances(name, loaded_model, X, Y)

def model_metrics(name, model, X_train, X_test, Y_train, Y_test, datatype):
	# Do all of the model metrics to save some room in script
	print("%s data results:\n" % datatype)

	# Fit model & make predictions
	fitted_model = model.fit(X_train, Y_train)
	Y_pred = fitted_model.predict(X_test)

	# Classification report & Confusion Matrix (needs separate training and evaluation process)
	classification_report(name, Y_test, Y_pred)
	confusion_matrix(name, Y_test, Y_pred)

	# Generate ROC Curves
	receiver_operating_characteristic(name, Y_test, Y_pred, datatype)

def stand_norm_test():
	# Another method of model fitting involves using standardized data
	# This works best for iterative models that converge to their fitted parameter values
	# I.e. approach the limits of parameter values as the iterations increase
	# Examples are: GLM, Neural Network, ...

	# Standardize data
	scaler = preprocessing.StandardScaler().fit(X)
	X_stand = scaler.transform(X)

	# Also normalize data
	X_norm = preprocessing.normalize(X)

	# Train test split: standardized
	X_train_stand, X_test_stand, Y_train_stand, Y_test_stand = split_train_test(X_stand, Y, 0.33, 142)

	# Train test split: normalized
	X_train_norm, X_test_norm, Y_train_norm, Y_test_norm = split_train_test(X_norm, Y, 0.33, 142)

	# Models
	models = np.empty([2, 2], dtype = 'object')
	models[0] = ['Logistic Regression', linear_model.LogisticRegression(random_state = 1)]
	models[1] = ['Neural Network', neural_network.MLPClassifier(random_state = 1, solver = 'adam', max_iter = 1000)]

	# Train models and report results
	for name, model in models:
		# First with standardized data
		# Different model metrics
		model_metrics(name, model, X_train_stand, X_test_stand, Y_train_stand, Y_test_stand, 'Standardized')
		for scoring in ('accuracy', 'roc_auc'):
			cross_validation(name, model, X_stand, Y, scoring)

		# Now with normalized data
		model_metrics(name, model, X_train_norm, X_test_norm, Y_train_norm, Y_test_norm, 'Normalized')
		for scoring in ('accuracy', 'roc_auc'):
			cross_validation(name, model, X_norm, Y, scoring)

def significant_variables():
	# Another potential next step would be to try fitting data only on the most significant covariates
	# In this case, variables 1, 5, 6, and 7 ("preg", "test", "mass", "pedi")
	# Let's fit some models using only these variables
	# We already have X & Y, but need to separate out the necessary variables
	X_sig = array[:, (0,4,5,6)]
	Y_sig = array[:, 8]

	# Train test split for evaluation metrics
	X_train_sig, X_test_sig, Y_train_sig, Y_test_sig = split_train_test(X_sig, Y_sig, 0.33, 142)

	n_trees = 100
	models = np.empty([2, 2], dtype = 'object')
	models[0] = ['Logistic Regression', linear_model.LogisticRegression(random_state = 1)]
	models[1] = ['Random Forest', ensemble.RandomForestClassifier(random_state = 1, n_estimators = n_trees, max_features = 3)]

	# Fit & evaluate models
	for name, model in models:
		# Different model metrics
		model_metrics(name, model, X_train_sig, X_test_sig, Y_train_sig, Y_test_sig, None)
		for scoring in ('accuracy', 'roc_auc'):
			cross_validation(name, model, X_sig, Y_sig, scoring)

	# Hmm accuracy and other metrics decrease slightly
	# But the Logistic Regression model keeps chugging along!



if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'spot':
            spot_check()
        elif sys.argv[1] == 'ensemble':
            ensemble_models()
        elif sys.argv[1] == 'voting':
            voting_ensemble()
        elif sys.argv[1] == 'final':
            final_models()
        elif sys.argv[1] == 'test':
        	stand_norm_test()
        elif sys.argv[1] == 'sig':
        	significant_variables()
        else:
        	print("Incorrect keyword; please enter a valid keyword.")
    else:
    	print("Incorrect number of keywords; please enter one keyword after the program name.")