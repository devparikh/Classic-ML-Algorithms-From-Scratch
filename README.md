# Classic-ML-Algorithms-From-Scratch
This reposity contains implementations of the following algorithms completely from scratch using only Numpy, Matplotlib, and Pandas.

1. K-Means Clustering
2. Random Forest
3. Naive Bayes

Let's understand how each of these algorithms work.

# K-Means Clustering:
K-Means Clustering is an algorithm that attempts to find the most optimal clusters for a given dataset by minimizing the within-cluster sum of distance. The within-cluster sum of distance is the distances from the centroid to every point in a given cluster. Optimizing to minimizing the within-cluster sum of distance, results in more compact clusters. More compact clusters means that there is a greater distance between the average points in 2 clusters(the centroids), this metric for finding the most optimal clusters is useful when using the KMeans++ Centroid Initialization technique.

# What does the process of training a K-Means Clustering algorithm look like?

1. Centroid Initialization - There are many different techniques for centroid initialization, although the most common is randomly selecting k points from the dataset to be initial centroids. My implementation of K-Means uses KMeans++ which attempts to maximize the distance between the initial centroids attempting to find the most optimal centroids. The KMeans++ approach saves lots of time & compute resources during training, as the model starts to train with optimal centroids. KMeans++ works by randomly initializing the first centroid, and for every other centroid finding a data point that is furthest away from the nearest initialized centroid.

2. Training K-Means Clustering
   1. Calculate the distance from each centroid to every point in the dataset, and add each point to the cluster of it's nearest centroid.
   2. Re-calculate the centroids for the clusters.
   3. Continue this process until the maximum iterations set are reached or until the centroids and points in the clusters do not change.

# What are the hyperparameters for K-Means Clustering:
- K value(Number of Clusters), to find the most optimal K value I used the Elbow Method
- The number of iterations to train K-Means on, although for my implementation I stopped training when the clusters stopped changing.

# Elbow Method:
For K-Means Clustering to find the most optimal K value, you can train the model for a range of K values and calculate the inertia for each configuration. The inertia is simply the total within-cluster sum of distance for every cluster in a configuration, and is a measure of the quality of clustered output. I noticed that if I can the script multiple times, the results weren't entirely consistent, so I decided to find the most optimal K value 100 times and K value that was most consistent throughout the test is choosen for training. 
