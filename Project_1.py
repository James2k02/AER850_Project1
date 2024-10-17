'''Importing Libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier


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
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns = X_train.columns[0:-1])
X_train = scaled_data_train_df.join(X_train.iloc[:,-1:])

scaled_data_test = my_scaler.transform(X_test.iloc[:,0:-1])
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns = X_test.columns[0:-1])
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

# ovr is one vs. rest which kind of turns it into a binary problem
my_model1 = LogisticRegression(C = 0.01, class_weight = 'balanced', multi_class='ovr', random_state = 69) 
my_model1.fit(X_train, y_train)
y_pred_train1 = my_model1.predict(X_train)
    
print("Classification Report for Train 1 \n", classification_report(y_train, y_pred_train1, zero_division = 0))

y_pred_test1 = my_model1.predict(X_test)

print("Classification Report for Test 1 \n", classification_report(y_test, y_pred_test1, zero_division = 0))
    
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

# max_iter is the maximum amount (limit) of iterations allowed for the optimization to converge (to find the optimal solution)
# 1) A higher value allows the model to run longer and potentially find a better solution but will have high computational costs
# 2) A lower value may lead to faster training but risks stopping the process before convergence
param_grid = {
    'C' : [0.01, 0.1, 1, 10, 100],
    'max_iter' : [100, 200, 300, 400, 500]
}

# scoring dictionary
# I use weighted metrics since there's an imbalance in the dataset where steps 7-9 have the most data points so that each class's
# contribution to the overall score is proportional to its size in the dataset

    
# grid search implementation where we refit based on f1_weighted (has balanced performance) to get a good balance between precision and recall
grid_search_model1 = GridSearchCV(estimator = my_model1, param_grid = param_grid, cv = 5, scoring = 'f1_weighted', n_jobs = 1)
grid_search_model1.fit(X_train, y_train)
best_params = grid_search_model1.best_params_
print("Best Hyperparameters for Model 1 (based on f1_weighted):", best_params)

best_model = grid_search_model1.best_estimator_
y_pred1 = best_model.predict(X_test)

# checking classification report to see how it predicted each individual class
print("Classification Report After GridSearchCV 1 \n", classification_report(y_test, y_pred1, zero_division = 0))

# confusion matrix of model 1
cm1 = confusion_matrix(y_test, y_pred1)
disp1 = ConfusionMatrixDisplay(confusion_matrix = cm1)
disp1.plot(cmap = plt.cm.Blues)
plt.title("Confusion Matrix for Logistic Regression")
plt.show()


# What I noticed is that the overall accuracy is very good at 94% but this can be misleading due to the fact that step 3 is very
# poorly predicted; the use of precision, recall, and f1 will give a better understanding of the metrics
# Some steps predicted better than others and some aren't predicted at all since it's being overfitted to the dominate classes

# Now, I will try some other models to see if this would get fixed since logistic regression is the most basic method of classification

  
# USING RANDOM FORESTS #

my_model2 = RandomForestClassifier(max_depth = 5, 
                                   n_estimators = 10,
                                   min_samples_split = 45, 
                                   min_samples_leaf = 40, 
                                   max_features = 'sqrt',
                                   class_weight = 'balanced', 
                                   random_state = 69)
my_model2.fit(X_train, y_train)
y_pred_train2 = my_model2.predict(X_train)
   
print("Classification Report for Train 2 \n", classification_report(y_train, y_pred_train2, zero_division = 0))

y_pred_test2 = my_model2.predict(X_test)

print("Classification Report for Test 2 \n", classification_report(y_test, y_pred_test2, zero_division = 0))
    
# GridSearchCV for Model 2

param_grid2 = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [30, 40, 50],
    'min_samples_leaf': [20, 30, 40],
    'max_features': ['sqrt'],
    'class_weight' : ['balanced']
}    

# n_estimators is the number of trees in the forest
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
grid_search_model2 = GridSearchCV(estimator = my_model2, param_grid = param_grid2, cv = 5, scoring = 'f1_weighted', n_jobs = 1)
grid_search_model2.fit(X_train, y_train)
best_params2 = grid_search_model2.best_params_

print("Best Hyperparameters for Model 2 (based on f1_weighted):", best_params2)

best_model2 = grid_search_model2.best_estimator_

y_pred2 = best_model2.predict(X_test)

print("Classification Report After GridSearchCV 2 \n", classification_report(y_test, y_pred2, zero_division = 0))

cm2 = confusion_matrix(y_test, y_pred2)

disp2 = ConfusionMatrixDisplay(confusion_matrix = cm2)
disp2.plot(cmap = plt.cm.Blues)
plt.title("Confusion Matrix for Random Forests")
plt.show()


# USING SVM #

my_model3 = SVC(class_weight = 'balanced', random_state = 69)
my_model3.fit(X_train, y_train)
y_pred_train3 = my_model3.predict(X_train)

print("Classification Report for Train 3 \n", classification_report(y_train, y_pred_train3, zero_division = 0))

y_pred_test3 = my_model3.predict(X_test)
print("Classification Report for Test 3 \n", classification_report(y_test, y_pred_test3, zero_division = 0))

# GridSearchCV for Model 3

param_grid3 = {
    'C': [0.1, 1, 5],          
    'kernel': ['linear', 'rbf', 'poly'], 
    'gamma': ['scale', 'auto'],  
    'class_weight' : ['balanced']      
}

# C is the regulatization parameter (same from logistic regression)

# Kernel determines how data is transformed into a higher-dimensional space
# 1) the right kernel choice can greatly affect performance

# Gamma affects the shape of the decision boundary and is particularly relevant when using non-linear kernals like rbf
# 1) defines how far the influence of a single transing example reachs (low values mean far-reaching influence and high values mean
# close influence)

# grid search implementation
grid_search_model3 = GridSearchCV(estimator = my_model3, param_grid = param_grid3, cv = 5, scoring = 'f1_weighted', n_jobs = 1)
grid_search_model3.fit(X_train, y_train)
best_params3 = grid_search_model3.best_params_

print("Best Hyperparameters for Model 3 (based on f1_weighted):", best_params3)

best_model3 = grid_search_model3.best_estimator_

y_pred3 = best_model3.predict(X_test)

print("Classification Report After GridSearchCV 3 \n", classification_report(y_test, y_pred3, zero_division = 0))

cm3 = confusion_matrix(y_test, y_pred3)

disp3 = ConfusionMatrixDisplay(confusion_matrix = cm3)
disp3.plot(cmap = plt.cm.Blues)
plt.title("Confusion Matrix for SVM")
plt.show()

# USING KNEIGHBORSCLASSIFIER #

my_model4 = KNeighborsClassifier()
my_model4.fit(X_train, y_train)
y_pred_train4 = my_model4.predict(X_train)

print("Classification Report for Train 4 \n", classification_report(y_train, y_pred_train4, zero_division = 0))

y_pred_test4 = my_model4.predict(X_test)
print("Classification Report for Test 4 \n", classification_report(y_test, y_pred_test4, zero_division = 0))

# RandomizedSearchCV for Model 4

param_grid4 = {
    'n_neighbors': [3, 5, 7, 9, 11],  
    'weights': ['uniform', 'distance'],  
    'p': [1, 2]  
}

# n_neighbors defines the number os nearest neighbors the algorithm will use to make a prediction for each point
# 1) Smaller values are sensitive to noise and could result in overfitting
# 2) Larger values makes the decision smoother but could potentially underfit the data

# weights determines how the neighbors contribute to the decision making process
# 1) uniform means all neighbors have equal weight
# 2) distance means closer neighbors have more influence than those further away

# p defines which distance metric will be used for the calculation
# 1) 1 = Manhattan distance; 2 = Euclidean distance
  
# randomized search implementation

random_search = RandomizedSearchCV(my_model4, param_grid4, n_iter = 10, cv = 5, scoring = 'f1_weighted', random_state = 69, n_jobs = -1)
random_search.fit(X_train, y_train)

best_params4 = random_search.best_params_
print("Best Hyperparameters for Model 4 (based on f1_weighted):", best_params4)

best_model4 = random_search.best_estimator_

y_pred4 = best_model4.predict(X_test)

print("Classification Report After RandomizedSearchCV \n", classification_report(y_test, y_pred4, zero_division = 0))

cm4 = confusion_matrix(y_test, y_pred4)

disp4 = ConfusionMatrixDisplay(confusion_matrix = cm4)
disp4.plot(cmap = plt.cm.Blues)
plt.title("Confusion Matrix for KNN")
plt.show()






















