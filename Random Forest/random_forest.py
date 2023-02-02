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
categorical_features = ["Pclass", "Sex", "Embarked"]

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

# A function for finding the best feature for splitting a parent node into children nodes
def perform_best_node_split(feature_set, X_data, y_data):
    global X_column, classes
    features_information_gain = []
    children_node_data = []
    for feature in feature_set:
        if feature in categorical_features:
            X_column =  X_data[feature]
            classes = sorted(list(X_column.unique()))

            children_node_y = [[] for _ in range(len(classes))]
            children_node_X = [[] for _ in range(len(classes))]
            
            child_node_counter = 0
            for category in classes:
                for feature_row in range(0, len(X_column)):
                    if X_column.iloc[feature_row] == category:
                        child_y_data = y_data.iloc[feature_row]
                        child_X_data = X_data.iloc[feature_row]

                        children_node_y[child_node_counter].append(child_y_data)
                        children_node_X[child_node_counter].append(child_X_data)   
    
                child_node_counter += 1 

            child_node_data = [children_node_X, children_node_y]
            children_node_data.append(child_node_data)
        
            # Calculating Information Gain
            information_gain_value = information_gain(y_data, children_node_y)
            features_information_gain.append(information_gain_value)
        
        else:
            X_column = X_data[feature]
            average_value = X_column.mean()
        
            n = 2
            children_node_y = [[], []]
            children_node_X = [[], []]
            for feature_row in range(0, len(X_column)): 
                if X_column[feature_row] <= average_value:
                    children_node_y[0].append(y_data.iloc[feature_row])
                    children_node_X[0].append(X_data.iloc[feature_row])

                else:
                    children_node_y[1].append(y_data.iloc[feature_row])
                    children_node_X[1].append(X_data.iloc[feature_row])

            child_node_data = [children_node_X, children_node_y]
            children_node_data.append(child_node_data)

            # Calculating Information Gain
            information_gain_value = information_gain(y_data, children_node_y)
            features_information_gain.append(information_gain_value)
    
    optimal_feature_index = features_information_gain.index(max(features_information_gain))
    optimal_feature = feature_set[optimal_feature_index]
    print(optimal_feature)

    optimal_feature_node_data = children_node_data[optimal_feature_index] 
    X_data = optimal_feature_node_data[0]
    y_data = optimal_feature_node_data[1]

    return X_data, y_data

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, node):
        self.children.append(node)

def building_decision_tree(feature_set, X_data, y_data):
    global nodes
    nodes = []
    depth = max_depth
    features = max_features
    while depth > 0 and len(y_dataset) >= min_sample_split and features > 0:
        # Create the parent node and add it's children nodes
        parent_node_data = [X_data, y_data]
        parent_node = Node(parent_node_data)
        children_X_data, children_y_data = perform_best_node_split(feature_set, X_data, y_data)

        for (child_node_X, child_node_y) in zip(children_X_data, children_y_data):
            child_node_data = [child_node_X, child_node_y]
            child_node = Node(child_node_data)
            parent_node.add_child(child_node)

        feature_set.remove(feature_set[optimal_feature_index])
        depth -= 1
        features -= 1

        nodes.append(parent_node)
