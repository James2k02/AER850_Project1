'''Importing Libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns



'''Identifying Objectives and Variables'''

# The main objective of this lab is to be able to predict with step an inverted will be at when given coordinate
# inputs X, Y, and Z

# This is a classification problem where the inputs are classified depending on the step it corresponds with; there
# will be 13 classes total with a certain number of coordinate sets in each


'''Data Processing'''

# using function to read data from .csv file and storing in df (DataFrame) variable
df = pd.read_csv("Project_1_Data.csv")

'''Data Visualization'''

# functions to visualize the raw data





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
