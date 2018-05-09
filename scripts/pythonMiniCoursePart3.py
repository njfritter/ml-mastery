#
##
###
#### PART 10: Extra in depth Analysis
###
##
#

# The process from the other parts of the mini course were satisfactory for a good model selection
# But there are other things that we could have done as well; let's do them here

import numpy as np
import pandas as pd
import sys
from sklearn import (preprocessing, model_selection, linear_model, 
	metrics, neighbors, neural_network, svm, ensemble
)
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

# Define url and columns
url = "https://goo.gl/vhm1eU"
columns = np.array(['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

data = pd.read_csv(url, names = columns)
array = data.values

# Separate into covariates and predictor
X = array[:, 0:8]
Y = array[:, 8]

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

def receiver_operating_characteristic(name, labels, predictions, datatype):
	# ROC Curves
	try:
		fpr, tpr, threshold = metrics.roc_curve(y_true = labels, y_score = predictions, pos_label = 1)
		roc_auc = metrics.auc(fpr, tpr)

		if datatype is not None:
			plt.title('%s ROC Curve: %s' % (datatype, name))
		else:
			plt.title('ROC Curve: %s' % (name))
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
	except: 
		print("The %s model is not compatible with the %s method" % (name, feature_importances_.__name__))

def split_train_test(X, Y, test_size, random_state):
	# Split data into training and testing
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
		X, Y, test_size = test_size, random_state = random_state)

	return X_train, X_test, Y_train, Y_test

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

def regression():
	# Now we will do a regression problem for one part
	# It deserves more, but this is all I can do for now
	# Most of the algorithms above can be used for regression
	# So I will use some of the ones I haven't used yet
	new_url = 'https://goo.gl/sXleFv'
	new_columns = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])

	dataframe = pd.read_csv(new_url, delim_whitespace = True, names = new_columns)
	array = dataframe.values
	X = array[:, 0:13]
	Y = array[:, 13]

	k_fold = model_selection.KFold(n_splits = 10, random_state = 7)
	models = np.empty([6, 2], dtype = 'object')
	models[0] = ['K Nearest Neighbors', neighbors.KNeighborsRegressor()]
	models[1] = ['Linear Regression', linear_model.LinearRegression()]
	models[2] = ['Ridge Regression', linear_model.Ridge()]
	models[3] = ['Support Vector Regressor', svm.LinearSVR()]
	models[4] = ['Random Forest Regressor', ensemble.RandomForestRegressor()]
	models[5] = ['Gradient Boosted Trees', ensemble.GradientBoostingRegressor()]

	for name, model in models:
		# Different model metrics
		for scoring in ('neg_mean_squared_error', 'explained_variance'):
			cross_validation(name, model, X, Y, scoring)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'test':
        	stand_norm_test()
        if sys.argv[1] == 'sig':
        	significant_variables()
        if sys.argv[1] == 'reg':
        	regression()


