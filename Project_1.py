'''Importing Libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit



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

































