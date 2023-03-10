# Classic-ML-Algorithms-From-Scratch
This reposity contains implementations of the following algorithms completely from scratch using only Numpy, Matplotlib, and Pandas.

1. K-Means Clustering
2. Random Forest
3. Naive Bayes

Let's understand how each of these algorithms work.

# K-Means Clustering:
K-Means Clustering is an algorithm that attempts to find the most optimal clusters for a given dataset by minimizing the within-cluster sum of distance. The within-cluster sum of distance is the distances from the centroid to every point in a given cluster. Optimizing to minimizing the within-cluster sum of distance, results in more compact clusters. More compact clusters means that there is a greater distance between the average points in 2 clusters(the centroids), this metric for finding the most optimal clusters is useful when using the KMeans++ Centroid Initialization technique.

**What does the process of training a K-Means Clustering algorithm look like?**

1. Centroid Initialization - There are many different techniques for centroid initialization, although the most common is randomly selecting k points from the dataset to be initial centroids. My implementation of K-Means uses KMeans++ which attempts to maximize the distance between the initial centroids attempting to find the most optimal centroids. The KMeans++ approach saves lots of time & compute resources during training, as the model starts to train with optimal centroids. KMeans++ works by randomly initializing the first centroid, and for every other centroid finding a data point that is furthest away from the nearest initialized centroid.

2. Training K-Means Clustering
   1. Calculate the distance from each centroid to every point in the dataset, and add each point to the cluster of it's nearest centroid.
   2. Re-calculate the centroids for the clusters.
   3. Continue this process until the maximum iterations set are reached or until the centroids and points in the clusters do not change.

**What are the hyperparameters for K-Means Clustering:**
- K value(Number of Clusters), to find the most optimal K value I used the Elbow Method
- The number of iterations to train K-Means on, although for my implementation I stopped training when the clusters stopped changing.

**Elbow Method:**
For K-Means Clustering to find the most optimal K value, you can train the model for a range of K values and calculate the inertia for each configuration. The inertia is simply the total within-cluster sum of distance for every cluster in a configuration, and is a measure of the quality of clustered output. I noticed that if I can the script multiple times, the results weren't entirely consistent, so I decided to find the most optimal K value 100 times and K value that was most consistent throughout the test is choosen for training. 

# Random Forest

**What is Random Forest and how does the algorithm work?** 

Random Forest is essentially just an ensemble model of decision trees, which aims to make a prediction about the independent variable(y data) given the features(x data) or dependent variables. During model evaluation, each tree makes a prediction given a single data point, and the most consistent prediction from the Random Forest is the output of the random forest model. This approach is in many instances more optimal than a singular decision tree making predictions, because the error of some trees can be covered up other trees that are making predictions in the right direction. This is why it’s important to try to build trees that are as uncorrelated from each other as possible, and that can only happen when the features & input training data have been randomly sampled. This is done through Bagging(Bootstrapped Aggregation) and Random Sampling for the feature set. Decision trees are also referred to as flowcharts, where each node & its data splits into some number of children nodes depending on the feature chosen for the split at that node.  The decision tree’s goal is to separate the input training y data into the distinct classes, or in other words to increase the uniformity of the data in every child node in the tree. The reason why the tree optimizes for this is because, the greater the uniformity of a node’s y data, the greater confidence it has about its prediction. Let’s put this into perspective to understand the importance of this when attempting to make predictions on the test set, let’s say that we have a node with some input data and the feature selected for the split is **Age**, and assuming that the mean age is 25 from all of the training data points. Then, the left node will have data where the age of people is less than 25, and the right node will have data where the age of the people is greater than 25. If the majority class in the left node is 1(survived) and in the right node is 0(not survived), we can deduce that if a person’s age is greater than 25, it can be confidently predicted that they are not going to survive. 

**How Decision Trees work:**

*How do we select a feature for a parent node to perform the split?*

For every parent node in the decision tree, you want to find a feature that will maximize the uniformity of the data in the children nodes. This results in the most optimal feature being selected for the first parent node and the last parent node would have the least optimal feature in the tree. 

**Entropy:**

![Untitled](Untitled.png)

The metric used in this implementation for measuring the uniformity of data of is as Entropy, which is the measure of disorder or uncertainity in a dataset. This means that a higher entropy value would mean greater disorder or less uniformity in data, making any predictions from it more uncertain. An entropy value of 0.0 in the case of 2 classes, would mean that the dataset is entirely uniform(completely 0 or 1), and on the other hand an entropy value of 1.0 would imply that the disorder/uncertainity in the dataset is at it’s peak, or that an equal number of elements are of each class in the data. Any entropy value between 0 and 1 would imply that there is more elements of a particular class than the other classes the data. 

![Screen Shot 2023-01-23 at 10.27.30 PM.png](Screen_Shot_2023-01-23_at_10.27.30_PM.png)

The input of the entropy function, *E* is the input *y* data, *S*. The function takes the negative percentage of elements of class *i* in comparison to the length of *S* and multiples this with the logarithm(base 2) of this percentage*.* This is performed for very class in the input data.

**Information Gain:**

The goal of the decision tree is to reduce the entropy value as nodes get split into children nodes, so the most optimal split would be when the difference between the parent node’s entropy and the weighted average of the children node’s entropy is maximized. This is what Information Gain is.

![Untitled](Untitled.png)

Information Gain is calculated by subtracting the weighted average entropy of the children nodes, where Y represents the parent node’s data, and Y|X represents the collective data of the children nodes. 

At every split in the tree, we are selecting the highest information gain for every feature remaining in the feature set. 

**How did I implement the algorithm step-by-step, and what were some interesting patterns/techniques and lessons I learnt throughout the process?**

**Implementation Process:**

1. Loading, Preprocessing and Balancing Data
2. Build a function to calculate *Entropy* and integrating that into another function for calculating *Information* *Gain*.
3. Creating n randomly sampled(with replacement) feature sets and training datasets for the decision trees.
4. Creating *perform_best_node_split* to determine the best possible feature for every split in a decision tree, and to make the split after the best split has been determined.
5. Building a Node Class, that contains the attributes, *data*(storing the X and y data during the training process), *children*(stores the memory locations of it’s children nodes), and *split_feature*(stores the feature that was used to split the node during training). These class attributes are integral in building the decision tree in the next step.
6. Creating function *building_decision_trees* to build a decision tree, which starts by inputting the input data into the root node and then iteratively performing some operations as long as terminal_node ≠ ****True and the feature set is not empty**.** 
    - In the loop, there are 2 parts, one is a set of operations that are performed for the first split in the tree, which is when the root node splits into the first children nodes. The other set of operations are performed for every other split in the tree until the terminal nodes are reached.
        - The only difference between the 2 sets of operations is that we only split the root node into children node for the in the first case, but when we are splitting nodes that are not root nodes, there are multiples nodes on every layer, where each node needs to be split into it’s children nodes.
7. Creating function random_forest to create n decision trees, and retrieve it’s respective feature set and training set.
8. Creating function *clear_decision_tree_data*, which traverses through each layer of the tree, clearing all of the data in every node except terminal nodes. For every terminal node in the tree, it calculates what the most frequently occurring class is, and that becomes the prediction of that node.
9. Create function *testing_random_forest*, which takes as input the testing X data, testing the Random Forest on one datapoint at a time. You are essentially using this data and the feature set to find the next node in the tree until you reach a terminal node, or a node with no children. Once a terminal node has been reached, the prediction is taken and stored along with the predictions of the other decision trees. The most consistent prediction from all of the trees is the prediction of the Random Forest. This process is performed for every datapoint in the test set.
10. Creating function *accuracy*, which does it what is says, it calculates the accuracy of the model’s predictions. 

K-Fold Cross-Validation:

K-Fold Cross-Validation is a method that I used to attempt to maximize the performance(accuracy was the metric) of the model by trying all of the different configurations of hyperparamters(length of each feature set, number of decision trees and the percentage of the entire dataset that is training data. 

*Process of performing K-Fold Cross Validation:*

1. Determining a K-Value(in this implementation K = 4)
2. Split the dataset into k sections/folds
3. Iteratively select every possible combination of the hyperparameters listed above. 
    - For each combination, iterate over each section/fold that have been created, and set it as the validation data and the rest of the data to be training data.
    - Take the average validation accuracy of all of the validation accuracies from the previous step, this is the accuracy of that specific configuration.
4. The optimality of the configuration is determined by this, by taking the configuration that had the highest average test accuracy.
5. Once the most optimal configuration has been determined, we use it to build the model, and test it on the pre-split test data. 

**What are the technical details of of my model, it’s performance on the train & test sets and techniques I used to improve it?**

Dataset:

The original dataset used for this project was the Titanic Survivors Prediction, with 891 rows of data(people), but this wasn’t the dataset that was split into the training & testing set. As there was an imbalance in the number of people(rows) that did not survived and the number of people who did survived, 342 and 549 respectively. This would negatively impact my ability to evaluate the performance of my model through a metric like accuracy. When there is a class imbalance, the model has an inherent bias towards the dominating class, with most of the terminal nodes having a prediction of that class. On top of this, this imbalance would be reflected in the test set as well, resulting in a misleading high accuracy that does not accurately show the model’s inability to generalize to new data.  

To counter act this, I sampled 684 rows from the original dataset, where the number of people who had survived were both 342.  One of the hyperparameters for the model was the training percentage, or the percentage of this data that would be a part of the training set. The different values were 0.7, 0.8, and 0.9 or 70%, 80%, 90%. 

Number of Features:

The total number of features that are used for training the model are 7 features; *Pclass, Age, Sex, SibSp, Parch, Fare, Embarked*. During the random feature sampling process, x features are randomly sampled(with replacement) from these 7 features. During the K-Fold Cross Validation, there are 2 possible values initialized for the number of features per feature set, 4 or 5 features. One of these numbers was chosen in the most optimal configurations of hyperparameters.

Number of Decision Trees:

This is one of the most important hyperparameters along with number of features, and is highly correlated with it as well. The reason for the success of ensemble models like Random Forest in comparison to Decision Trees is that the error made by trees can be covered by the other trees whose predictions are accurate, resulting in the forest making an accurate prediction as long as the majority of trees are correct. This requires trees to be as uncorrelated as possible. Picking a large number of features, would result in correlated trees. In this case have a large number of trees is not advantageous, and can be counterproductive as the same errors would be echoed throughout. On the contrary, picking a smaller number of features would mean more uncorrelated trees, where a larger number of trees would be optimal. 

The right balance between the number of features chosen for each feature set, and the number of decision trees is required to maximize accuracy.

Minimum Number of Samples For Split:

This wasn’t one of the hyperparameters that would be chosen, as it was already pre-determined to be 25. Essentially, this is one parameter that is used to determine when to stop splitting nodes into children nodes, because every time splits are performed the amount of data in the children nodes decreases, and after a certain point there isn’t enough data in the nodes to be split further, so there needs to be a cut off point where a node without enough data becomes the terminal node. 

In this implementation this parameter was trivial, because it didn’t have a significant impact on the depth of the trees, that was almost entirely dependent on the length of the feature set.
