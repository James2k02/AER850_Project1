'''Importing Libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


'''Identifying Objectives and Variables'''

# The main objective of this project is to be able to predict with step an inverted will be at when given coordinate
# inputs X, Y, and Z

# This is a classification problem where the inputs are classified depending on the step it corresponds with; there
# will be 13 classes total with a certain number of coordinate sets in each


'''Data Processing'''

# using function to read data from .csv file and storing in df (DataFrame) variable
df = pd.read_csv("Project_1_Data.csv")


'''Data Visualization'''

# functions to visualize the raw data
# will be using class distributions to check where all the data is located and if it's even between all the steps
step_count = df["Step"].value_counts()
step_count.plot(kind = "bar")
plt.title("Step Class Distribution")
plt.xlabel("Step")
plt.ylabel("No. of Instances")
plt.show()


# By examining the bar graph for the number of instances in each step, it is noted that in steps 7, 8, and 9,
# there is a lot more data points in those steps which indicates that we need to use StratifiedShuffleSplit for
# our train/test data split because there is a data imbalance which might predict results very accurately for those steps
# but will achieve very poor results when predicting other steps

# StratifiedShuffleSplit ensures that the train/test data maintains the same proportions of each class in the dataset
# which will avoid bias towards steps 7, 8, and 9


'''Data Split into Train/Test'''

# using StratifiedShuffleSpit function to perform the train/test data split
my_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 69)

# for every index that we take for the split, we want to reset the index
for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)

        
'''Variable Selection'''

# selecting output by dropping tht column from the x_train and x_test and selecting it for the y_train and y_test
X_train = strat_df_train.drop("Step", axis = 1)
y_train = strat_df_train["Step"]
X_test = strat_df_test.drop("Step", axis = 1)
y_test = strat_df_test["Step"]

'''Data Cleaning and Preprocessing'''

# SCALING #

# Using df.describe() I was able to see different statistics to determine whether or not I should implement scaling
# The std of the 3 features are quite different with X being much more spread out compared to Z so X might have more influence
# Even though the values are not extremely far apart, it still might affect the performance of the model so it would
# be in the best interest of the model to perform scaling methods
my_scaler = StandardScaler()
my_scaler.fit(X_train.iloc[:,0:-1])
scaled_data_train = my_scaler.transform(X_train.iloc[:,0:-1])
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns=X_train.columns[0:-1])
X_train = scaled_data_train_df.join(X_train.iloc[:,-1:])

scaled_data_test = my_scaler.transform(X_test.iloc[:,0:-1])
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns=X_test.columns[0:-1])
X_test = scaled_data_test_df.join(X_test.iloc[:,-1:])

# CORRELATION MATRIX #

# getting correlation matrix for the inputs 
corr_matrix = X_train.corr()

# heatmap to visualize the correlation matrix
sns.heatmap(np.abs(corr_matrix))

# by examining the heatmap, we see that the all the inputs have very low colinearities(highest ~0.2) relative 
# to each other which means we do not need to drop any variables

'''Model Training and Cross Validation'''

# USING LOGISTIC REGRESSION #

my_model1 = LogisticRegression(multi_class='ovr') # ovr is one vs. rest which kind of turns it into a binary problem
my_model1.fit(X_train, y_train)
y_pred_train1 = my_model1.predict(X_test)

for i in range(5):
    print("Predictions:", y_pred_train1[i], "Actual values:", y_train[i])
    
# GridSearchCV for Model 1 

# should only perform grid search on train dataset

# defining the hyperparameter grid

# C is the regularization variable which refers to the amount of penalty applied to the model's complexity 
# 1) Regularization helps prevent overfitting (low train error and high test error) by discouraging the model from assigning too much
#    importance to any one feature which makes it more generalizable to unseen data
# 2) A large C value (weak regularization) allows the model to have more flexibility (less penalization) meaning it can fit more complex
#    patterns and assign larger coefficients to features but increases the risk of overfitting
# 3) A small C value (strong regularization) forces the model to shrink the coefficients and make the model simpler and reduces 
#    overfitting but potenially underiftting if C is too small

# max_iter is the maximum amount (limit) of iterations allowed for the optimization to converge (to find the optimum solution)
# 1) A higher value allows the model to run longer and potentially find a better solution but will have high computational costs
# 2) A lower value may lead to faster training but risks stopping the process before convergence
param_grid = {
    'C' : [0.01, 0.1, 1, 10, 100],
    'max_iter' : [100, 200, 300, 400, 500]
}

# scoring dictionary
# I use weighted metrics since there's an imbalance in the dataset where steps 7-9 have the most data points so that each class's
# contribution to the overall score is proportional to its size in the dataset
scoring = {
    'accuracy': 'accuracy',
    'f1_weighted': 'f1_weighted',
    'precision_weighted' : 'precision_weighted',
    'recall_weighted' : 'recall_weighted'
}
    
# grid search implementation where we refit based on f1_weighted (has balanced performance) to get a good balance between precision and recall
grid_search_model1 = GridSearchCV(estimator = my_model1, param_grid = param_grid, cv = 5, scoring = scoring, refit = 'f1_weighted', n_jobs = 1)
grid_search_model1.fit(X_train, y_train)
best_params = grid_search_model1.best_params_
print("Best Hyperparameters for Model 1 (based on f1_weighted):", best_params)

best_model = grid_search_model1.best_estimator_

# checking training set accuracy
train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
print("Training Set Accuracy:", train_accuracy)

# checking test set accuracy
y_pred1 = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred1)
print("Test Set Accuracy", test_accuracy)

# checking classification report to see how it predicted each individual class
print(classification_report(y_test, y_pred1))

# What I noticed is that the overall accuracy is very good at 94% but this can be misleading due to the fact that step 3 is very
# poorly predicted; the use of precision, recall, and f1 will give a better understanding of the metrics
# Some steps predicted better than others, for instance, class 10 and 11 have much lower scores than the other classes
# Step 3 completely failed to predict

# Now, I will try some other models to see if this would get fixed since logistic regression is the most basic method of classification

  
# USING RANDOM FORESTS #

my_model2 = RandomForestClassifier(random_state=69)
my_model2.fit(X_train, y_train)
y_pred_train2 = my_model2.predict(X_test)

for i in range(5):
    print("Predictions:", y_pred_train2[i], "Actual values:", y_train[i])
    
# GridSearchCV for Model 2

param_grid2 = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}    

# n_estimators is the number os trees in the forest
# 1) increasing this generally improves performance because there would be more variance but at some point it will just be computationally
#    expensive and take long to solve

# max_depth is the min number of samples required to split an internal node
# 1) higher values make the tree more conservative which prevents it from splitting too much and overfitting
# 2) smaller numbers allow the tree to be more complex

# min_samples_split is the minimum number of samples needed to split an internal node
# 1) higher values = more conservation (reduced overfitting)

# min_samples_leaf is the the minimum number of samples required to be at a leaf node
# 1) higher values = more conservation (reduced overfitting)

# max_features is the number of features to consider when looking for the best split
# 1) controls how many features each tree is allowed to consider when splitting a node
# 2) reducing the number of features can help reduce overfitting

# grid search implementation
grid_search_model2 = GridSearchCV(estimator = my_model2, param_grid = param_grid2, cv = 5, scoring = scoring, refit = 'f1_weighted', n_jobs = 1)
grid_search_model2.fit(X_train, y_train)
best_params2 = grid_search_model2.best_params_

print("Best Hyperparameters for Model 2 (based on f1_weighted):", best_params2)

best_model2 = grid_search_model2.best_estimator_

print(classification_report(y_test, best_model2.predict(X_test)))


# USING SVM #

my_model3 = SVC(random_state = 69)
my_model3.fit(X_train, y_train)
y_pred_train3 = my_model3.predict(X_test)

for i in range(5):
    print("Predictions:", y_pred_train3[i], "Actual values:", y_train[i])
    
# GridSearchCV for Model 3

param_grid3 = {
    'C': [0.1, 1, 10, 100],          
    'kernel': ['linear', 'rbf', 'poly'], 
    'gamma': ['scale', 'auto']         
}

# C is the regulatization parameter (same from logistic regression)

# Kernel determines how data is transformed into a higher-dimensional space
# 1) the right kernel choice can greatly affect performance

# Gamma affects the shape of the decision boundary and is particularly relevant when using non-linear kernals like rbf
# 1) defines how far the influence of a single transing example reachs (low values mean far-reaching influence and high values mean
# close influence)

# grid search implementation
grid_search_model3 = GridSearchCV(estimator = my_model3, param_grid = param_grid3, cv = 5, scoring = scoring, refit = 'f1_weighted', n_jobs = 1)
grid_search_model3.fit(X_train, y_train)
best_params3 = grid_search_model3.best_params_

print("Best Hyperparameters for Model 3 (based on f1_weighted):", best_params3)

best_model3 = grid_search_model3.best_estimator_

print(classification_report(y_test, best_model3.predict(X_test)))


# USING RANDOMIZEDSEARCHCV #































