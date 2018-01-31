#
##
###
#### PART 10: Extra in depth Analysis
###
##
#


# The process from the other parts of the mini course were satisfactory for a good model selection
# But there are other things that we could have done as well
# Let's do them here

import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import ensemble

# Define url and columns
url = "https://goo.gl/vhm1eU"
columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

def part_ten():
	# Another method of model fitting involves using standardized data
	# This works best for iterative models that converge to their fitted parameter values
	# I.e. approach the limits of parameter values as the iterations increase
	# Examples are: GLM, Neural Network, ...
	# So let's train those
	data = pd.read_csv(url, names = columns)
	array = data.values

	# Separate into covariates and predictor
	X = array[:, 0:8]
	Y = array[:, 8]

	# Standardize data
	scaler = preprocessing.StandardScaler().fit(X)
	X_stand = scaler.transform(X)

	# Also normalize data
	X_norm = preprocessing.normalize(X)

	# Train test split: standardized
	X_train_stand, X_test_stand, Y_train_stand, Y_test_stand = model_selection.train_test_split(
		X_stand, Y, test_size = 0.33, random_state = 42)	

	# Train test split: normalized
	X_train_norm, X_test_norm, Y_train_norm, Y_test_norm = model_selection.train_test_split(
		X_norm, Y, test_size = 0.33, random_state = 142)	

	# Models
	models = []
	models.append(('Logistic Regression', linear_model.LogisticRegression(random_state = 1)))
	models.append(('Neural Network', neural_network.MLPClassifier(random_state = 1, solver = 'adam', max_iter = 500)))

	# Train models and report results
	for name, model in models:
		# First with standardized data
		# Accuracy
		scoring = 'accuracy'
		k_fold = model_selection.KFold(n_splits = 10, random_state = 1)
		result = model_selection.cross_val_score(model, X_stand, Y, cv = k_fold, scoring = scoring)
		print("\n Accuracy of %s model with standardized data:\n %.3f%% (+\-%.3f%%)" % (name, result.mean() * 100.0, result.std() * 100.0))

		# ROC values
		scoring = 'roc_auc'
		result = model_selection.cross_val_score(model, X_stand, Y, cv = k_fold, scoring = scoring)
		print("\n ROC value of %s model with standardized data:\n %.3f%% (+\-%.3f%%)" % (name, result.mean(), result.std()))

		# Confusion Matrix and Classification report
		fitted_model = model.fit(X_train_stand, Y_train_stand)
		Y_pred_stand = model.predict(X_test_stand)
		conf_matrix_stand = metrics.confusion_matrix(Y_test_stand, Y_pred_stand)
		class_report_stand = metrics.classification_report(Y_test_stand, Y_pred_stand)
		print("\nConfusion matrix for %s model with standardized data:\n" % (name), conf_matrix_stand)
		print("\nClassification report for %s model with standardized data:\n" % (name), class_report_stand)

		# Now with normalized data
		# Accuracy
		scoring = 'accuracy'
		k_fold = model_selection.KFold(n_splits = 10, random_state = 1)
		result = model_selection.cross_val_score(model, X_norm, Y, cv = k_fold, scoring = scoring)
		print("\n Accuracy of %s model with normalized data:\n %.3f%% (+\-%.3f%%)" % (name, result.mean() * 100.0, result.std() * 100.0))

		# ROC values
		scoring = 'roc_auc'
		result = model_selection.cross_val_score(model, X_norm, Y, cv = k_fold, scoring = scoring)
		print("\n ROC value of %s model with normalized data:\n %.3f%% (+\-%.3f%%)" % (name, result.mean(), result.std()))

		# Confusion Matrix and Classification report
		fitted_model = model.fit(X_train_norm, Y_train_norm)
		Y_pred_norm = model.predict(X_test_norm)
		conf_matrix_norm = metrics.confusion_matrix(Y_test_norm, Y_pred_norm)
		class_report_norm = metrics.classification_report(Y_test_norm, Y_pred_norm)
		print("\nConfusion matrix for %s model with normalized data:\n" % (name), conf_matrix_norm)
		print("\nClassification report for %s model with normalized data:\n" % (name), class_report_norm)


def part_eleven():
	# Now we will do a regression problem for one part
	# It deserves more, but this is all I can do for now
	# Most of the algorithms above can be used for regression
	# So I will use some of the ones I haven't used yet
	new_url = 'https://goo.gl/sXleFv'
	new_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

	dataframe = pd.read_csv(new_url, delim_whitespace = True, names = new_columns)
	array = dataframe.values
	X = array[:, 0:13]
	Y = array[:, 13]

	k_fold = model_selection.KFold(n_splits = 10, random_state = 7)
	models = []
	models.append(('K Nearest Neighbors', neighbors.KNeighborsRegressor()))
	models.append(('Linear Regression', linear_model.LinearRegression()))
	models.append(('Ridge Regression', linear_model.Ridge()))
	models.append(('Support Vector Regressor', svm.LinearSVR()))
	models.append(('Random Forest Regressor', ensemble.RandomForestRegressor()))
	models.append(('Gradient Boosted Trees', ensemble.GradientBoostingRegressor()))

	for name, model in models:
		# Negative mean squared error
		scoring = 'neg_mean_squared_error'
		results = model_selection.cross_val_score(model, X, Y, cv = k_fold, scoring = scoring)
		print("\nNeg Mean Squared Error of %s model:" % (name), results.mean())
		print("\nStandard deviation NMSE of %s model:" % (name), results.std())

		# Something else...
		scoring = 'explained_variance'
		results = model_selection.cross_val_score(model, X, Y, cv = k_fold, scoring = scoring)
		print("\nExplained Variance of %s model:" % (name), results.mean())
		print("\nStandard deviation of Explained Variance of %s model:" % (name), results.std())

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == '10':
        	part_ten()
        if sys.argv[1] == '11':
        	part_eleven()


