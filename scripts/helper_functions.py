#!/usr/bin/env python3

# Mastering ML Python Mini Course
# Inspired by the project here: 
# https://s3.amazonaws.com/MLMastery/machine_learning_mastery_with_python_mini_course.pdf?__s=mxhvphowryg2sfmzus2q
# By Nathan Fritter
# Project will soon be found at: 
# https://www.inertia7.com/projects/

################

# Import Packages

# Here we will define helper functions
# Which we will import into some other scripts
# Those scripts focus on specific parts of the course
# So here will be the source of all the functions
import numpy as np
import pandas as pd
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

# Separate into input and output components
X = array[:, 0:8]
Y = array[:, 8]	

# Train test split for evaluation metrics
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
X, Y, test_size = 0.33, random_state = 42)

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


def load_data():

	# Method 1
	read_data(url, 1)

	# Method 2
	read_data(url, 2)

	# Method 3
	read_data(url, 3)

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
	X = array[:, (0,4,5,6)]
	Y = array[:, 8]

	# Train test split for evaluation metrics
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
	X, Y, test_size = 0.33, random_state = 42)

	n_trees = 100
	models = np.empty([2, 2], dtype = 'object')
	models[0] = ['Logistic Regression', linear_model.LogisticRegression(random_state = 1)]
	models[1] = ['Random Forest', ensemble.RandomForestClassifier(random_state = 1, n_estimators = n_trees, max_features = 3)]

	# Fit & evaluate models
	for name, model in models:
		# Different model metrics
		model_metrics(name, model, X_train, X_test, Y_train, Y_test, None)
		for scoring in ('accuracy', 'roc_auc'):
			cross_validation(name, model, X, Y, scoring)


def regression():
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