# Implementing K-Means Clustering from Scratch

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random 
# Loading the Mall Customers Data CSV File
df = pd.read_csv("Mall_Customers.csv")

# Converting the Categorical Values of Gender to 1 for Male and 0 for Female
df["Gender"] = df["Gender"].replace("Male", 1)
df["Gender"] = df["Gender"].replace("Female", 0)

# For simplicity we are just going to take 2 dimensions(for easier visualization & clustering) from the 4 dimensions, with one indepedent variable and the other dependent variable 
# Indepedent Variable - Annual Income in (k$)
# Dependent Variable - Spending Score (1-100)

input_dataframe = df[["Annual Income (k$)", "Spending Score (1-100)"]]

x = df["Annual Income (k$)"]
y = df["Spending Score (1-100)"]

'''Building K-Means Clustering Model'''

# Model Parameters
number_of_clusters = 5

def centroid_initialization(k, input_data):
    # Centroid Initialization through KMeans++
    global points
    global centroids
    points = []

    for index in input_data.index:
        x_y_pair = []
        # Extracting the x and y values from each row in the input data

        x = input_data["Annual Income (k$)"][index]
        y = input_data["Spending Score (1-100)"][index]
        x_y_pair.append(x)
        x_y_pair.append(y)
        points.append(x_y_pair)
    
    # Randomly Choosing the First Centroid
    centroids = []
    centroid = random.choice(points)
    centroids.append(centroid)
    points.remove(centroid)

    while len(centroids) <= k - 1:
        distance_from_nearest_point = []
        for point in points:
            point_nearest_centroid = [] 
            for centroid in centroids:
                distance_squared = (point[0] - centroid[0])**2 + (point[1] - centroid[1])**2
                point_nearest_centroid.append(distance_squared)

            nearest_centroid = min(point_nearest_centroid)
            distance_from_nearest_point.append(nearest_centroid)

        centroid = points[distance_from_nearest_point.index(max(distance_from_nearest_point))]
        centroids.append(centroid)
        points.remove(centroid)

centroid_initialization(number_of_clusters, input_dataframe)


def K_Means_Clustering(k, centroids, input_data, iterations=100):
    clusters = []
    i = 0
    while i < 5:
        i += 1
        clusters.append([])

    for point in points:
        distance_to_centroids = []
        for centroid in centroids:                                                                                                                                                                                                                      
            distance_squared = (centroid[0] - point[0])**2 + (centroid[1] - point[1])**2
            distance_to_centroids.append(distance_squared)

        nearest_cluster = clusters[distance_to_centroids.index(min(distance_to_centroids))]
        nearest_cluster.append(point)
