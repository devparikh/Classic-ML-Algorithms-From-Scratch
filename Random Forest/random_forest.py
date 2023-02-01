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

# Filling in missing values in the Age Column with the mode(age with the greatest frequency in the column) age
input_dataframe["Age"].fillna(input_dataframe["Age"].mode()[0], inplace=True)
input_dataframe["Embarked"].fillna(input_dataframe["Embarked"].mode()[0], inplace=True)

# Convert the Embarked and Sex column into a numerical column for ease of use
input_dataframe["Sex"].replace(["male", "female"], [1, 0], inplace=True)
input_dataframe["Embarked"].replace(["S", "Q", "C"], [1, 2, 3], inplace=True)

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

# This function calculated the information gain given a specific feature that would split the parent node into children node(s)
def information_gain(parent_node_classes, children_y):
    # Calculating Information Gain
    parent_entropy = entropy(parent_node_classes)

    children_node_weighted_entropys = []
    for data in children_y:
        child_node_entropy = entropy(data)
        child_node_weighted_entropy = float(len(data) / len(parent_node_classes) * child_node_entropy)
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

    return feature_sets

def bagging(input_dataset, x):
    n_value = 0
    X_bootstrapped_datasets = []
    y_bootstrapped_datasets = []

    while n_value < x:
        # Bagging(Bootstrap Aggregating) is a statistical method where we can sample with replacement n new datasets of length x from an dataset of length x
        bootstrapped_dataset = input_dataset.sample(n=len(input_dataset))

        X_bootstrapped_dataset = bootstrapped_dataset.drop("Survived", axis=1)
        y_bootstrapped_dataset = bootstrapped_dataset["Survived"]

        X_bootstrapped_datasets.append(X_bootstrapped_dataset)
        y_bootstrapped_datasets.append(y_bootstrapped_dataset)

        n_value += 1

    return X_bootstrapped_datasets, y_bootstrapped_datasets

'''Building the Random Forest'''

# Hyperparams for Random Forest 
max_depth = 6
max_features = 4
min_sample_split = 100
min_samples_leaf = 99
num_of_decision_trees = 100

# A function for finding the best feature for splitting a parent node into children nodes
def find_best_split_feature(feature_set, X_data, y_data):
    categorical_features = ["Pclass", "Sex", "Embarked"]
    features_information_gain = []
    for feature in feature_set:
        if feature in categorical_features:
            X_column =  X_data[feature]
            classes = sorted(list(X_column.unique()))

            children_node_y = [[] for _ in range(len(classes))]
            child_node_counter = 0

            for category in classes:
                for feature_row in range(0, len(X_column)):
                    if X_column.iloc[feature_row] == category:
                        child_y_data = y_data.iloc[feature_row]

                        children_node_y[child_node_counter].append(child_y_data)

                child_node_counter += 1

            # Calculating Information Gain
            information_gain_value = information_gain(y_data, children_node_y)

            features_information_gain.append(information_gain_value)
        
        else:
            X_column = X_data[feature]
            average_value = X_column.mean()
        
            n = 2
            children_node_y = [[] for _ in range(n)]
            for feature_row in range(0, len(X_column)): 
                if X_column[feature_row] <= average_value:
                    children_node_y[0].append(y_data.iloc[feature_row])

                else:
                    children_node_y[1].append(y_data.iloc[feature_row])

            # Calculating Information Gain
            information_gain_value = information_gain(y_data, children_node_y)
            
            features_information_gain.append(information_gain_value)

    optimal_feature = feature_set[features_information_gain.index(max(features_information_gain))]
    return optimal_feature
