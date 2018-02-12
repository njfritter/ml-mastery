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
from sklearn import (preprocessing, model_selection, linear_model, 
	metrics, neighbors, neural_network, svm, ensemble
	)

# Define url and columns
url = "https://goo.gl/vhm1eU"
columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(url, names = columns)
array = data.values

# Separate into covariates and predictor
X = array[:, 0:8]
Y = array[:, 8]

def part_ten():
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
		# Different model metrics
		print("Standardized data results:\n")
		for scoring in ('accuracy', 'roc_auc'):
			k_fold = model_selection.KFold(n_splits = 10, random_state = 1)
			try:
				result = model_selection.cross_val_score(model, X_stand, Y, cv = k_fold, scoring = scoring)
			except AttributeError:
				print("The %s model cannot perform cross validation with the %s metric" % (name, scoring))
			if scoring == 'accuracy':
				print("\n%s of %s model:\n %.3f%% (+\-%.3f%%)" 
				% (scoring, name, result.mean() * 100.0, result.std() * 100.0))
			else:
				print("\n%s of %s model:\n %.3f (+\-%.3f)" % (scoring, name, result.mean(), result.std()))	


		# Confusion Matrix and Classification report
		fitted_model = model.fit(X_train_stand, Y_train_stand)
		Y_pred_stand = model.predict(X_test_stand)
		conf_matrix_stand = metrics.confusion_matrix(Y_test_stand, Y_pred_stand)
		class_report_stand = metrics.classification_report(Y_test_stand, Y_pred_stand)
		print("\nConfusion matrix for %s model with standardized data:\n" % (name), conf_matrix_stand)
		print("\nClassification report for %s model with standardized data:\n" % (name), class_report_stand)

		# Now with normalized data
		print("Normalized data results:\n")
		for scoring in ('accuracy', 'roc_auc'):
			k_fold = model_selection.KFold(n_splits = 10, random_state = 1)
			try:
				result = model_selection.cross_val_score(model, X_norm, Y, cv = k_fold, scoring = scoring)
			except AttributeError:
				print("The %s model cannot perform cross validation with the %s metric" % (name, scoring))
			if scoring == 'accuracy':
				print("\n%s of %s model:\n %.3f%% (+\-%.3f%%)" 
				% (scoring, name, result.mean() * 100.0, result.std() * 100.0))
			else:
				print("\n%s of %s model:\n %.3f (+\-%.3f)" % (scoring, name, result.mean(), result.std()))	

		# Confusion Matrix and Classification report
		fitted_model = model.fit(X_train_norm, Y_train_norm)
		Y_pred_norm = model.predict(X_test_norm)
		conf_matrix_norm = metrics.confusion_matrix(Y_test_norm, Y_pred_norm)
		class_report_norm = metrics.classification_report(Y_test_norm, Y_pred_norm)
		print("\nConfusion matrix for %s model with normalized data:\n" % (name), conf_matrix_norm)
		print("\nClassification report for %s model with normalized data:\n" % (name), class_report_norm)


def part_eleven():
	# Another potential next step would be to try fitting data only on the most significant covariates
	# In this case, variables 1, 5, 6, and 7 ("preg", "test", "mass", "pedi")
	# Let's fit some models using only these variables
	# We already have X & Y, but need to separate out the necessary variables
	X = array[:, (0,4,5,6)]
	Y = array[:, 8]

	# Train test split for evaluation metrics
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
	X, Y, test_size = 0.33, random_state = 42)

	n_trees = 100
	models = []
	models.append(('Logistic_Regression', linear_model.LogisticRegression(random_state = 1)))
	models.append(('Random_Forest', ensemble.RandomForestClassifier(random_state = 1, n_estimators = n_trees, max_features = 3)))

	# Fit & evaluate models
	for name, model in models:
		# Different model metrics
		for scoring in ('accuracy', 'roc_auc'):
			k_fold = model_selection.KFold(n_splits = 10, random_state = 1)
			try:
				result = model_selection.cross_val_score(model, X, Y, cv = k_fold, scoring = scoring)
			except AttributeError:
				print("The %s model cannot perform cross validation with the %s metric" % (name, scoring))
			if scoring == 'accuracy':
				print("\n%s of %s model:\n %.3f%% (+\-%.3f%%)" 
				% (scoring, name, result.mean() * 100.0, result.std() * 100.0))
			else:
				print("\n%s of %s model:\n %.3f (+\-%.3f)" % (scoring, name, result.mean(), result.std()))	

		# Classification report & Confusion Matrix (need to do separate training and evaluation process)
		fitted_model = model.fit(X_train, Y_train)
		Y_pred = model.predict(X_test)
		conf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
		class_report = metrics.classification_report(Y_test, Y_pred)
		print("\nConfusion Matrix for %s model:\n" % (name), conf_matrix)
		print("\nClassification Report for %s model:\n" % (name), class_report)

		# ROC Curves
		try:
			Y_prob = model.predict(X_test)
			fpr, tpr, threshold = metrics.roc_curve(y_true = Y_test, y_score = Y_prob, pos_label = 1)
			roc_auc = metrics.auc(fpr, tpr)
			
			plt.title('Receiver Operating Characteristic')
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
			print("The %s model does not support the \"predict\" method" % name)


	# Hmm accuracy and other metrics decrease slightly
	# But the Logistic Regression model keeps chugging along!

def part_twelve():
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
		# Different model metrics
		for scoring in ('neg_mean_squared_error', 'explained_variance'):
			results = model_selection.cross_val_score(model, X, Y, cv = k_fold, scoring = scoring)
			print("\nMean %s of %s model:" % (scoring, name), results.mean())
			print("\nStandard deviation %s of %s model:" % (scoring, name), results.std())


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == '10':
        	part_ten()
        if sys.argv[1] == '11':
        	part_eleven()
        if sys.argv[1] == '12':
        	part_twelve()


