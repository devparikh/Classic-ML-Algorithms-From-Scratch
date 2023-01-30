# Implementing Random Forest from Scratch 

# Importing libraries for this implementation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import numpy as np

'''Loading Data + Data Preprocessing'''
input_dataframe = pd.read_csv("train.csv")

# Filling in missing values in the Age Column with the mode(age with the greatest frequency in the column) age, and the missing data in the Cabin column is fileld with NaN values
# The mean and median values effect the outcome o
input_dataframe.fillna({'Cabin': 'NaN'})
input_dataframe["Age"].fillna(input_dataframe["Age"].mode()[0], inplace=True)

# Convert the column Sex into a numerical column for ease of use
input_dataframe["Sex"].replace(["male", "female"], [1, 0], inplace=True)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

input_dataframe = input_dataframe.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Train Test Split
# 90% is training and 10% is testing
train_test_percentage_split = int(0.9 * len(input_dataframe))

training_data = input_dataframe.iloc[:train_test_percentage_split, :]
testing_data = input_dataframe.iloc[train_test_percentage_split:, :]

'''Implementing Random Forests'''

# Creating functions for calculating entropy and information gain, key metrics for building effective decision trees
# Entropy measures the uniformity of data in a given node or the measure of impurity of the data
# Information Gain measures the decrease in impurity/uniformity of data from a parent node to it's children nodes

def entropy(class_distribution_data):
    global Entropy
    survived = []
    not_survived = []
    for output_label in class_distribution_data:
        if output_label == 0:
            not_survived.append(output_label)
        else:
            survived.append(output_label)

    survived_probability = len(survived) / len(class_distribution_data)
    not_survived_probability = len(not_survived) / len(class_distribution_data)

    Entropy = -(survived_probability * math.log2(survived_probability)) - (not_survived_probability * math.log2(not_survived_probability))
    return Entropy

def categorical_features_data_split(feature, parent_column_data, parent_output_data):
    classes = parent_column_data.unique()

    children_node_data = []
    for category in classes:
        category_classes = []
        for row in range(0, len(parent_column_data)):
            if parent_column_data[row] == category:
                category_classes.append(parent_output_data[row])

        children_node_data.append(category_classes)

    return children_node_data

def continuous_features_data_split(parent_column_data, parent_output_data):
    # For features like Age, SipSp, Parch, Fare where there are not distinct categories rather numerical values 
    # The approach is to split the parent node into children nodes by finding the average value of that feature

    average_value = parent_column_data.mean()

    n = 2
    children_node_data = [[] for _ in range(n)]
    for row in range(0, len(parent_column_data)):
        if parent_column_data[row] <= average_value:
                children_node_data[0].append(parent_output_data[row])

        else:
            children_node_data[1].append(parent_output_data[row])

    return children_node_data   

# This function calculated the information gain given a specific feature that would split the parent node into children node(s)
def information_gain(feature, parent_column_data, parent_node_classes):
    categorical_features = ["Pclass", "Sex", "Embarked"]

    if feature in categorical_features:
        children_node_data = categorical_features_data_split(feature, parent_column_data, parent_node_classes)
          
    else:
        children_node_data = continuous_features_data_split(parent_column_data, parent_node_classes)
    
    # Calculating Information Gain
    parent_entropy = entropy(parent_node_classes)

    children_node_weighted_entropys = []
    for data in children_node_data:
        child_node_entropy = entropy(data)
        child_node_weighted_entropy = len(data) / len(parent_node_classes) * child_node_entropy
        children_node_weighted_entropys.append(child_node_weighted_entropy)

    weighted_average_entropy = sum(children_node_weighted_entropys)

    information_gain = parent_entropy - weighted_average_entropy
    return information_gain

def random_features_sampling(input_features, n_feature_sets, n_features):
    # This function samples n elements from a feature set for x new feature sets
    num_feature_sets = 0
    feature_sets = []
    while num_feature_sets < n_feature_sets:
        feature_set = random.sample(input_features, n_features)
        feature_sets.append(feature_set)
        num_feature_sets += 1

def bagging(input_dataset, x):
    n_value = 0
    X_bootstrapped_datasets = []
    y_bootstrapped_datasets = []

    while n_value < x:
        # Bagging(Bootstrap Aggregating) is a statistical method where we can sample with replacement n new datasets of length x from an dataset of length x
        bootstrapped_dataset = input_dataset.sample(n=len(input_dataset))

        X_bootstrapped_dataset = bootstrapped_dataset.drop("Survived", axis=1)
        y_bootstrapped_dataset = bootstrapped_dataset["Survived"]

        X_bootstrapped_array = X_bootstrapped_dataset.to_numpy()
        y_bootstrapped_array = y_bootstrapped_dataset.to_numpy()

        X_bootstrapped_datasets.append(X_bootstrapped_array)
        y_bootstrapped_datasets.append(y_bootstrapped_array)

        n_value += 1
