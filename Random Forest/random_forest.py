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
            # If the feature is categorial, this is performed:

            X_column =  X_data[feature]

            # Getting all categories for a specific column in inout data
            classes = sorted(list(X_column.unique()))

            children_node_y = []
            children_node_X = []
            
            # Appending every row in the X_data that has the same value as the category from the input dataset into a new dataframe
            # Taking the y_data(predictions) of each row selected and storing them together, creating new X and y data for a child node
            for category in classes:
                X_dataframe = pd.DataFrame(columns=features)
                y_list = []
                for feature_row in range(0, len(X_column)):
                    if X_column.iloc[feature_row] == category:
                        # Storing the predictions(0s and 1s) in a list format
                        child_y_data = y_data[feature_row]
                        y_list.append(child_y_data)

                        child_X_data = X_data.iloc[feature_row]
                        child_X_data = {'Pclass' : child_X_data["Pclass"], 'Sex' : child_X_data["Sex"], 'Age' : child_X_data["Age"], 'SibSp' : child_X_data["SibSp"], 'Parch': child_X_data["Parch"], 'Fare' : child_X_data["Fare"], 'Embarked' : child_X_data["Embarked"]}
                        X_dataframe = X_dataframe.append(child_X_data, ignore_index=True)

                # Storing the X and y datasets for n number of nodes, where n is the number of classes/categories of the choosen feature
                children_node_X.append(X_dataframe)
                children_node_y.append(y_list)

            # Concatenated the data
            child_node_data = [children_node_X, children_node_y]
            children_node_data.append(child_node_data)
        
            # Calculating Information Gain
            information_gain_value = information_gain(y_data, children_node_y)
            features_information_gain.append(information_gain_value)
        
        else:
            # If the feature column has non-categorical values
            X_column = X_data[feature]

            # Calculating the average value of the feature in the dataset
            average_value = X_column.mean()
        
            children_node_y = []
            children_node_X = []

            # When a non-categorical feature is choosen for splitting a node, it gets split into 2 nodes, the left node is for all data points that are equal to or lower than the average value calculated in the feature's column
            # The rest of the data points go to the right node

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

    # After calculating the information gain for every feature in the feature set for the current node, the feature with the highest information gain is selected
    optimal_feature_index = features_information_gain.index(max(features_information_gain))
    optimal_feature = feature_set[optimal_feature_index]

    optimal_feature_node_data = children_node_data[optimal_feature_index] 
    X_data = optimal_feature_node_data[0]
    y_data = optimal_feature_node_data[1]

    return X_data, y_data, optimal_feature

'''Building out the Random Forest'''

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []
    
    def add_child(self, node):
        self.children.append(node)

def building_decision_tree(feature_set, X_data, y_data, depth, terminal_node=False):
    # Concatenating the X & y data to create the input data to be passed to the root node
    input_dataset = [X_data, y_data]
    # Creating the root node using the Node object initialize above this function
    root_node = Node(input_dataset)

    # Initializing 2 lists used in the loop below
    children_node_data = []
    children_nodes = []

    features_used = []

    iteration = 0
    while terminal_node == False and len(feature_set) > 0:
        if iteration == 0:
            # For the first split in the tree, we will split from the singular root node using only 1 feature
            children_X_data, children_y_data, child_optimal_feature = perform_best_node_split(feature_set, X_data, y_data)

            for (child_node_X, child_node_y) in zip(children_X_data, children_y_data):
                child_node_data = [child_node_X, child_node_y]
                child_node = Node(child_node_data)

                # Appending the data of the node being passed to the child node
                children_node_data.append(child_node_data)
                # Adding the child node to be referenced later for further splitting
                children_nodes.append(child_node)

                # Appending each child node to the (parent node).children method referenced later when traversing through the tree
                root_node.add_child(child_node)

            feature_set.remove(child_optimal_feature)
            features_used.append(child_optimal_feature)

            depth -= 1
            iteration += 1

        elif iteration >= 1:
            # Setting the 
            parent_nodes_data = children_node_data
            parent_nodes = children_nodes

            children_node_data = []
            children_nodes = []
            
            # Iterating over each parent node(leaf nodes of the current tree), attempting to split them further into n children nodes
            for index in range(0, len(parent_nodes_data)):
                # Getting the data for parent nodes
                parent_node_data = parent_nodes_data[index]
                parent_X_data = parent_node_data[0]
                parent_y_data = parent_node_data[1]

                parent_node = parent_nodes[index]

                parent_entropy = entropy(parent_y_data)

                # Performing the splitting until there are no features left in the feature set & num of samples meets the minimum requirement of 50 rows(data points)
                if len(feature_set) > 0 and len(parent_y_data) >= min_sample_split and parent_node != 0:
                    parent_node = parent_nodes[index]

                    children_X_data, children_y_data, child_optimal_feature = perform_best_node_split(feature_set, parent_X_data, parent_y_data)
                    feature_set.remove(child_optimal_feature)
                    features_used.append(child_optimal_feature)

                    for (child_node_X, child_node_y) in zip(children_X_data, children_y_data):
                        child_node_data = [child_node_X, child_node_y]
                        child_node = Node(child_node_data)
                        children_node_data.append(child_node_data)
                        children_nodes.append(child_node)
                        parent_node.add_child(child_node)

                # if the conditions are not meet, the parent nodes become leaf nodes and the tree is built
                else:
                    terminal_node = True

        else:
            terminal_node = True

    return root_node, features_used

def random_forest(features_sets, X_dataset, y_dataset, depth):
    trees = []
    features = []
    for index in range(0, len(features_sets)):
        X_data = X_dataset[index]
        y_data = y_dataset[index]
        features_set = features_sets[index]

        root_node, features_used = building_decision_tree(features_set, X_data, y_data, depth)
        trees.append(root_node)
        features.append(features_used)

    return trees, features

# Parameters for Random Forest
max_depth = 4
min_sample_split = 50

# Sampling feature sets and X and y data for building the trees
X_dataset, y_dataset = bagging(input_dataframe, x=100)
features_sets = random_features_sampling(features, n_feature_sets=100, n_features=4)

# Building the trees
trees, features_set = random_forest(features_sets, X_dataset, y_dataset, max_depth)

# Traversing through the trees
tree_num = 1
for index in range(0, len(trees)):
    root_node = trees[index]
    features = features_set[index]

    print("-----Tree #{}------".format(tree_num))

    children_nodes = root_node.children
    print("Layer #1")
    print("-------------------")
    print("Split #1: {}".format(features[0]))
    print("-------------------")
    print("Root Node")

    layer = 1

    while len(children_nodes) > 0:
        parent_nodes = children_nodes
        children_nodes = []

        print("Layer #{}".format(layer))
        print("-------------------")
        print("Split #{} Feature: {}".format(layer, features[layer]))
        print("-------------------")

        node_counter = 0
        for parent_node in parent_nodes:
            node_counter += 1
            print("Child Node #{}".format(node_counter))

            child_nodes = parent_node.children
            if len(child_nodes) > 0:
                for child_node in child_nodes:
                    children_nodes.append(child_node)

        layer += 1

    tree_num += 1
