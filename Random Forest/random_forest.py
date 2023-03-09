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

def input_data_balancing(input_dataframe, dataframe_2, dataframe_1, has_second_dataframe, class_max):
    # Train Test Split
    survived = 0
    not_survived = 0

    for index_value in range(len(input_dataframe)):
        column_value = input_dataframe.loc[index_value, "Survived"]
        row = input_dataframe.iloc[index_value]

        if column_value == 0 and not_survived < class_max:
            not_survived += 1
            dataframe_1 = dataframe_1.append(row, ignore_index=True)

        elif column_value == 1 and survived < class_max:
            survived += 1
            dataframe_1 = dataframe_1.append(row, ignore_index=True)

        else:
            if has_second_dataframe == True:
                dataframe_2 = dataframe_2.append(row, ignore_index=True)

    return dataframe_1, dataframe_2

df1 = pd.DataFrame(columns=features)   
df2 = pd.DataFrame(columns=features) 

data, other_data = input_data_balancing(input_dataframe, df1, df2, has_second_dataframe=False, class_max=342)

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

    for data_index in range(0, len(children_y)):
        data = children_y[data_index]
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

        y_bootstrapped_dataset = bootstrapped_dataset["Survived"]
        X_bootstrapped_dataset = bootstrapped_dataset.drop("Survived", axis=1)

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
                for feature in range(0, len(X_column)):
                    if X_column.iloc[feature] == category:
                        # Storing the predictions(0s and 1s) in a list format
                        child_y_data = y_data[feature]
                        y_list.append(child_y_data)

                        child_X_data = X_data.iloc[feature]
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

            left_X_dataframe = pd.DataFrame(columns=features)
            right_X_dataframe = pd.DataFrame(columns=features)

            left_y_list = []
            right_y_list = []
            for feature in range(0, len(X_column)): 
                if X_column.iloc[feature] <= average_value:
                    child_y_data = y_data[feature]
                    left_y_list.append(child_y_data)

                    child_X_data = X_data.iloc[feature]
                    child_X_data = {'Pclass' : child_X_data["Pclass"], 'Sex' : child_X_data["Sex"], 'Age' : child_X_data["Age"], 'SibSp' : child_X_data["SibSp"], 'Parch': child_X_data["Parch"], 'Fare' : child_X_data["Fare"], 'Embarked' : child_X_data["Embarked"]}
                    left_X_dataframe = left_X_dataframe.append(child_X_data, ignore_index=True)
                    
                else:
                    child_y_data = y_data[feature]
                    right_y_list.append(child_y_data)

                    child_X_data = X_data.iloc[feature]
                    child_X_data = {'Pclass' : child_X_data["Pclass"], 'Sex' : child_X_data["Sex"], 'Age' : child_X_data["Age"], 'SibSp' : child_X_data["SibSp"], 'Parch': child_X_data["Parch"], 'Fare' : child_X_data["Fare"], 'Embarked' : child_X_data["Embarked"]}
                    right_X_dataframe = right_X_dataframe.append(child_X_data, ignore_index=True)

            children_node_X.append(left_X_dataframe)
            children_node_X.append(right_X_dataframe)

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
    def __init__(self, data, split_feature=None):
        self.data = data
        self.children = []
        self.split_feature = split_feature
    
    def add_child(self, node):
        self.children.append(node)

def building_decision_tree(feature_set, X_data, y_data, terminal_node=False):
    # Concatenating the X & y data to create the input data to be passed to the root node
    input_dataset = [X_data, y_data]
    # Creating the root node using the Node object initialize above this function
    root_node = Node(input_dataset)

    # Initializing 2 lists used in the loop below
    children_node_data = []
    children_nodes = []

    iteration = 0
    while terminal_node == False and len(feature_set) > 0:
        if iteration == 0:
            # For the first split in the tree, we will split from the singular root node using only 1 feature
            children_X_data, children_y_data, child_optimal_feature = perform_best_node_split(feature_set, X_data, y_data)
            root_node.split_feature = child_optimal_feature

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

            iteration += 1

        elif iteration >= 1:
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
                if len(feature_set) > 0 and len(parent_y_data) >= min_sample_split and parent_entropy != 0:
                    parent_node = parent_nodes[index]

                    children_X_data, children_y_data, child_optimal_feature = perform_best_node_split(feature_set, parent_X_data, parent_y_data)
                    parent_node.split_feature = child_optimal_feature

                    feature_set.remove(child_optimal_feature)

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

    return root_node

def random_forest(features_sets, X_dataset, y_dataset):
    trees = []
    for index in range(0, len(features_sets)):
        X_data = X_dataset[index]
        y_data = y_dataset[index]
        features_set = features_sets[index]

        root_node = building_decision_tree(features_set, X_data, y_data)
        trees.append(root_node)

    return trees

def clear_decision_tree_data(trees):
    # Clearing training data from each node of the decision trees
    for root_node in trees:
        # Clearing the root node's data and finding the first set of children nodes

        root_node.data = []
        children_nodes = root_node.children

        while len(children_nodes) > 0:
            # setting the children nodes to parent nodes as they are going to be split further
            parent_nodes = children_nodes
            # reinitialize children_nodes to store the new child nodes
            children_nodes = [] 

            for parent_node in parent_nodes:
                # get the children of the current node and append them to children_nodes as long as (current_node).children is not empty
                child_nodes = parent_node.children
                if len(child_nodes) > 0:
                    parent_node.data = []
                    for child_node in child_nodes:
                        children_nodes.append(child_node)
                else:
                    parent_y_data = parent_node.data[1]
                    parent_node.data = int(max(set(parent_y_data), key = parent_y_data.count))

def testing_random_forest(trees, X_data):
    preds = []
    for index in range(0, len(X_data)):
        row = X_data.iloc[index]

        tree_preds = []
        for index in range(0, len(trees)):
            node = trees[index]
            while len(node.children) > 0:
                feature = node.split_feature
                children_nodes = node.children
                                 
                if feature in categorical_features:
                    row_feature_value = int(row[feature])

                    if len(children_nodes) == 2 and feature == "Sex":
                        classes = [0, 1]
                        node = children_nodes[classes.index(row_feature_value)]
                    elif len(children_nodes) == 3 and feature == "Embarked" or feature == "Pclass":
                        classes = [1, 2, 3]
                        node = children_nodes[classes.index(row_feature_value)]

                    else:
                        node = children_nodes[0]
                        
                else:
                    feature_column = X_data[feature]
                    average_value = feature_column.mean()

                    row_feature_value = row[feature]
                    
                    if row_feature_value <= average_value:
                        index = 0
                    else:
                        index = 1
                    
                    node = children_nodes[index]

            tree_pred = node.data
            tree_preds.append(tree_pred)

        random_forest_pred = int(max(set(tree_preds), key = tree_preds.count))
        preds.append(random_forest_pred)
    
    return preds

def accuracy(preds, ground_truth):
    accurate_pred = 0
    for index in range(len(ground_truth)):
        pred = preds[index]
        if ground_truth[index] == pred:
            accurate_pred += 1

    accuracy = int((accurate_pred / len(ground_truth)) * 100)
    return accuracy

# Performing K-Fold Cross Validation using different parameters to find the configuration with the greatest validation accuracy
k = 4
min_sample_split = 25

# Parameters for Random Forest
num_features_range = [4, 5, 6]
num_decision_trees_range = [25, 50, 75, 100]
training_percentage_range = [0.7, 0.8, 0.9]

configurations = []
test_accuracies = []

for num_features in num_features_range:
        for num_decision_trees in num_decision_trees_range: 
            for training_percentage in training_percentage_range:
                configuration = [num_features, num_decision_trees, training_percentage]
                configurations.append(configuration)

configuration_iteration = 0 
for configuration in configurations:
    configuration_iteration += 1
    print("The number of configurations considered thus far is", configuration_iteration)
    training_data = pd.DataFrame(columns=features)
    testing_data = pd.DataFrame(columns=features)

    training_length = int(configuration[-1] * len(data))
    training_data, testing_data = input_data_balancing(data, testing_data, training_data, has_second_dataframe=True, class_max=239)

    quarter_length = int(0.25 * len(training_data))

    first_fold = training_data.iloc[:quarter_length, :]
    second_fold = training_data.iloc[quarter_length:(2 * quarter_length), :]
    third_fold = training_data.iloc[(2 * quarter_length):(3 * quarter_length), :]
    fourth_fold = training_data.iloc[(3 * quarter_length):, :]

    k_fold_data = [first_fold, second_fold, third_fold, fourth_fold]

    average_test_accuracy = 0
    for fold in range(0, k):
        # Creating train and test set for this iteration of the cross_validation for each 
        test_data = k_fold_data[fold]
        train_data = pd.DataFrame(columns=features)
            
        for dataset in range(0, len(k_fold_data)):
            if dataset != fold:
                train_data = train_data.append(k_fold_data[dataset], ignore_index=True)

        test_ground_truth = test_data["Survived"].tolist()
        test_X_data = test_data.drop("Survived", axis=1)

        # Sampling feature sets and X and y data for building the trees 
        X_dataset, y_dataset = bagging(train_data, x=num_decision_trees)
        features_sets = random_features_sampling(features, n_feature_sets=num_decision_trees, n_features=num_features)
 
        # Building the trees
        trees = random_forest(features_sets, X_dataset, y_dataset)

        # Clear the decision and root node data
        clear_decision_tree_data(trees)

        preds = testing_random_forest(trees, test_X_data)

        test_accuracy = accuracy(preds, test_ground_truth) 
        average_test_accuracy += test_accuracy
        
    average_test_accuracy = round(average_test_accuracy / k)
    print("This configurations average test accuracy is", average_test_accuracy)
    test_accuracies.append(average_test_accuracy)

most_optimal_hyperparam_config = configurations[test_accuracies.index(max(test_accuracies))]

num_features = most_optimal_hyperparam_config[0]
num_decision_trees = most_optimal_hyperparam_config[1]
training_length = most_optimal_hyperparam_config[2]

training_data = pd.DataFrame(columns=features)
testing_data = pd.DataFrame(columns=features)
training_length = int(training_length * len(input_dataframe))

training_data, testing_data = input_data_balancing(input_dataframe, training_data, testing_data, has_second_dataframe=True, class_max=(training_length / 2))

X_dataset, y_dataset = bagging(training_data, x=num_decision_trees)
features_sets = random_features_sampling(features, n_feature_sets=num_decision_trees, n_features=num_features)

trees = random_forest(features_sets, X_dataset, y_dataset)
clear_decision_tree_data(trees)

test_ground_truth = testing_data["Survived"]
test_X_data = testing_data.drop("Survived", axis=1)

test_preds = testing_random_forest(trees, test_X_data)

test_accuracy = accuracy(test_preds, test_ground_truth)
print("The accuracy of the most optimal configuration of Random Forest is {} percent on {} test samples".format(test_accuracy, len(test_ground_truth)))

# The average testing accuracy from running 4 instances of this model was 91%, where 3 instances had 193 testing samples and the other had 267 testing samples.
# The individual testing accuracies were 100%, 98%, 88%, 78%
