# Python Machine Learning Mini Course

## Provided by Jason @ ML Mastery

### Introduction

My name is Nathan Fritter and I am a data geek. I have a B.S. in Applied Statistics from UCSB
and was an active member of the Data Science club at UCSB. I made numerous projects, presented
my findings and also was a member on the executive board. Feel free to view other projects
from the club as well as ones that I have completed for classes.

### The Course & Data
This project is inspired by the content [here](https://s3.amazonaws.com/MLMastery/machine_learning_mastery_with_python_mini_course.pdf?__s=mxhvphowryg2sfmzus2q).

This mini-course dealt mainly with the "Pimas Indians Diabetes" dataset from the 
UCI Machine Learning Repository (the raw data from Jason Brownlee's github can be found 
[here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv); unfortunately it does not look like the data is available on the UCI website [due to permission restrictions](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/)). Luckily, this Kaggle website happened to have more information on it [here](https://www.kaggle.com/uciml/pima-indians-diabetes-database). 

One of the parts also used the Boston Housing Data, which can be found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/)

The data has various fields relating to health (age, number of pregnancies the patient has had, Body Mass Index (BMI), insulin level, etc.), and in the course we use these fields to attempt to predict the target variable: an indicator variable showing 0 if the patient did not end up with diabetes, and 1 if they did. Since the target variable has only two outcomes (diabetes & no diabetes), this is a "classification" problem and the models used are tailored as such.

I have to point out that I am lucky that there was documentation of the data listed somewhere else. Domain knowledge (knowledge about the data due to first hand experience) is an important part of data analysis, so any other projects I do will make sure to do this. If anyone reading this has any supplemental information on the data, please let me know. 

I have added in my own comments, methods, and explanations of my thought process. 
Feel free to clone the repo, make a branch, leave comments or contribute in any way you'd like.

### Methods Used
This course had us dipping our feet in many different methods (models) for classification. I added in some as well that I wanted to try out. Here are the base models:

+ Logistic Regression
+ Linear Discriminant Analysis
+ Naive Bayes
+ Decision Tree
+ Neural Network
+ Stochastic Gradient Descent
+ Support Vector Machine
+ Ridge Classifier

I employed some "ensemble" methods as well (combinations of single base models above; more explanation [here](https://blog.statsbot.co/ensemble-learning-d1dcd548e936)):

+ AdaBoost Classifier
+ Gradient Boosted Machine
+ Random Forest Classifier
+ Extra Trees Classifier


### Steps to Reproduction

+ Clone the repo onto your machine (instructions [here](https://help.github.com/articles/cloning-a-repository/) if you are not familiar)
+ Download the package `virtualenv`
	+ Assuming you have made it this far you should have experience with one of `pip/pip3`, `brew` or another package manager
	+ Here we will use `pip3` (python3 compatible)
+ Run the following commands:
	+ `virtualenv venv` 
		+ This creates a virtual environment called "venv"
		+ Feel free to change the second word to whatever you'd like to call it
	+ `source venv/bin/activate` 
		+ This turns on the virtual environment and puts you in it
	+ `pip3 install -r requirements.txt`
		+ Installs every python package listed in the requirements.txt file
	+ `python3 pythonMiniCoursePart#.py *text*`
		+ Examine the code at the bottom of each part
		+ Replace # with whichever part you'd like to run
		+ Replace *text* with whichever part you'd like to run
		+ E.g. "spot" to spot check algorithms, "ensemble" to do ensemble methods, etc.

### Conclusions

Once I employed the methods, I saw that similar ones (Random Forest and Extra Trees, Ridget Regression and SGD, etc.) achieved similar results (of course) so I removed the lesser known ones. 

Of the simpler base models, the best performing ones were:
+ Logistic Regression
+ Linear Discriminant Analysis
+ Ridge Regression

All of the ensemble methods achieved pretty much the same accuracy as the base models, so I decided to pick a base model (Logistic Regression) and an ensemble method (Random Forest). 

A general rule is that you want to pick the simpler model due to training time; however, since this data is small and the difference in training time is negligible, I went ahead and used one of each as the final choices for the model.

In real life applications, the Logistic Regression model would be the one to go with. 


### Contributing

If you would like to contribute:
+ Create a new branch to work on (instructions [here](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches))
	+ By default the branch you will receive after cloning is the "master" branch
	+ This is where the main changes are deployed, so it should not contain any of your code unless the changes have been agreed upon by all parties (which is the purpose of creating the new branch
	+ This way you must submit a request to push changes to the main branch)
+ Submit changes as a pull request with any relevant information commented
	+ Reaching out in some way would also be great to help speed up the process

If you do contribute, I will add your name as a contributor!

### Iterations

This is a work in progress; I initally split up everything into parts depending on the minicourse.

However I will be splitting up by the different applications used, as this is better python practice.

*Update:* I have now split up the code into different parts and have dedicated functions to doing different tasks. 

Please let me know if there is a better way of going about things.

*Update 2:* So I have tried added titles to each of the plots in Part 1, yet alot are not centered.

If anyone knows why this is happening/would like to fix it please let me know or clone and make a pull request.

*Update 3:* I have overdone the try/except statements intentionally, so I will be scaling this back at some point. 

Also have just replaced all usage of lists and dictionaries with numpy arrays (more efficient storage).

*Update 4:* I will add graphs and charts to illustrate the differences between the methods used later.

### Sources Cited

+ Brownlee, Jason. Machine Learning Mastery with Python Mini-Course [https://s3.amazonaws.com/MLMastery/machine_learning_mastery_with_python_mini_course.pdf?__s=mxhvphowryg2sfmzus2q] *From Developer to Python Machine Learning Practitioner in 14 days*
+ Brownlee, Jason. Original Python Mini Course Github [https://github.com/JasonLian/Machine-Learning/blob/master/!Book%20and%20Paper/Machine_learning_mastery_with_Python_mini_course/Machine%20Learning%20with%20Python/mini_course.py]
+ Scikit-Learn Documentation [http://scikit-learn.org/stable/index.html]
+ Smolyakov, Vadim. Ensemble Learning to Improve Machine Learning Results. [https://blog.statsbot.co/ensemble-learning-d1dcd548e936]

