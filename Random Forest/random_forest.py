import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import numpy as np

import warnings

warnings.filterwarnings("ignore")

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

    if survived_probability == 1.0 or not_survived_probability == 1.0:
        Entropy = 0
    else:
        Entropy = -(survived_probability * math.log2(survived_probability)) - (not_survived_probability * math.log2(not_survived_probability))
    
    return Entropy

# This function calculated the information gain given a specific feature that would split the parent node into children node(s)
def information_gain(parent_node_classes, children_y):
    # Calculating Information Gain
    parent_entropy = entropy(parent_node_classes)

    for data in children_y:
        child_node_entropy = entropy(data)

        children_node_weighted_entropys = []
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

            children_node_y = []
            children_node_X = []
            
            for category in classes:
                X_dataframe = pd.DataFrame(columns=features)
                y_list = []
                for feature_row in range(0, len(X_column)):
                    if X_column.iloc[feature_row] == category:
                        child_y_data = y_data[feature_row]
                        y_list.append(child_y_data)

                        child_X_data = X_data.iloc[feature_row]
                        child_X_data = {'Pclass' : child_X_data["Pclass"], 'Sex' : child_X_data["Sex"], 'Age' : child_X_data["Age"], 'SibSp' : child_X_data["SibSp"], 'Parch': child_X_data["Parch"], 'Fare' : child_X_data["Fare"], 'Embarked' : child_X_data["Embarked"]}
                        X_dataframe = X_dataframe.append(child_X_data, ignore_index=True)

                children_node_X.append(X_dataframe)
                children_node_y.append(y_list)

            child_node_data = [children_node_X, children_node_y]
            children_node_data.append(child_node_data)
        
            # Calculating Information Gain
            information_gain_value = information_gain(y_data, children_node_y)
            features_information_gain.append(information_gain_value)
        
        else:
            X_column = X_data[feature]
            average_value = X_column.mean()
        
            children_node_y = []
            children_node_X = []

            left_child_X_dataframe = pd.DataFrame(columns=features)
            right_child_X_dataframe = pd.DataFrame(columns=features)

            left_y_list = []
            right_y_list = []
            for feature_row in range(0, len(X_column)): 
                if X_column.iloc[feature_row] <= average_value:
                    child_y_data = y_data[feature_row]
                    left_y_list.append(child_y_data)

                    child_X_data = X_data.iloc[feature_row]
                    child_X_data = {'Pclass' : child_X_data["Pclass"], 'Sex' : child_X_data["Sex"], 'Age' : child_X_data["Age"], 'SibSp' : child_X_data["SibSp"], 'Parch': child_X_data["Parch"], 'Fare' : child_X_data["Fare"], 'Embarked' : child_X_data["Embarked"]}
                    left_child_X_dataframe = left_child_X_dataframe.append(child_X_data, ignore_index=True)
                    
                else:
                    child_y_data = y_data[feature_row]
                    right_y_list.append(child_y_data)

                    child_X_data = X_data.iloc[feature_row]
                    child_X_data = {'Pclass' : child_X_data["Pclass"], 'Sex' : child_X_data["Sex"], 'Age' : child_X_data["Age"], 'SibSp' : child_X_data["SibSp"], 'Parch': child_X_data["Parch"], 'Fare' : child_X_data["Fare"], 'Embarked' : child_X_data["Embarked"]}
                    right_child_X_dataframe = right_child_X_dataframe.append(child_X_data, ignore_index=True)
            

            children_node_X.append(left_child_X_dataframe)
            children_node_X.append(right_child_X_dataframe)

            children_node_y.append(left_y_list)
            children_node_y.append(right_y_list)
            
            child_node_data = [children_node_X, children_node_y]
            children_node_data.append(child_node_data)

            # Calculating Information Gain
            information_gain_value = information_gain(y_data, children_node_y)
            features_information_gain.append(information_gain_value)

    optimal_feature_index = features_information_gain.index(max(features_information_gain))
    optimal_feature = feature_set[optimal_feature_index]

    optimal_feature_node_data = children_node_data[optimal_feature_index] 
    X_data = optimal_feature_node_data[0]
    y_data = optimal_feature_node_data[1]

    return X_data, y_data, optimal_feature

terminal_node = False

input_dataset = [X_data, y_data]
tree = Node(input_dataset)
layer = 1

iteration = 0
while terminal_node == False:
    if iteration == 0:
        children_X_data, children_y_data, child_optimal_feature = perform_best_node_split(features_set, X_data, y_data)

        for (child_node_X, child_node_y) in zip(children_X_data, children_y_data):
            child_node_data = [child_node_X, child_node_y]
            child_node = Node(child_node_data)
            tree.add_child(child_node, input_dataset)

        max_features -= 1

        iteration += 1

        layer += 1

    elif iteration > 0 and iteration < max_depth:
        if layer == 2:
            parent_nodes = tree.children
            layer += 1     
        
        else:   
            depth_previous_nodes = len(parent_nodes) + 1
            parent_nodes = []
            for children_node in range(depth_previous_nodes, len(tree.children)):  
                children_node = tree.children[children_node]
                parent_nodes.append(children_node)

            layer += 1
        
        for parent_node in parent_nodes:
            parent_X_data = parent_node.data[0]
            parent_y_data = parent_node.data[1]

            parent_data = [parent_X_data, parent_y_data]
            children_X_data, children_y_data, child_optimal_feature = perform_best_node_split(features_set, parent_X_data, parent_y_data)

            for (child_node_X, child_node_y) in zip(children_X_data, children_y_data):
                child_node_data = [child_node_X, child_node_y]
                child_node = Node(child_node_data)
                tree.add_child(child_node, parent_data)

        max_features -= 1

        iteration += 1

    else:
        terminal_node = True
