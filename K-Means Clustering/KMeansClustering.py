# Implementing K-Means Clustering from Scratch

# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd
import random 
# Loading the Mall Customers Data CSV File
df = pd.read_csv("Mall_Customers.csv")

# For simplicity we are just going to take 2 dimensions(for easier visualization & clustering) from the 4 dimensions, with one indepedent variable and the other dependent variable 
# Indepedent Variable - Annual Income in (k$)
# Dependent Variable - Spending Score (1-100)

input_dataframe = df[["Annual Income (k$)", "Spending Score (1-100)"]]

x = df["Annual Income (k$)"]
y = df["Spending Score (1-100)"]

'''Building K-Means Clustering Model'''

# The most optimal number of clusters for this dataset is 6
# This is computed by finding the number of clusters that results in the largest inertia value
number_of_clusters = 5

def Centroid_Initialization(k, input_data):
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

    # Initializing the rest of the centroids for KMeans
    while len(centroids) <= k - 1:
        distance_from_nearest_point = []
        # Iterating over each point, finding it's nearest cluster from all initialized centroids
        # Then, finding the point furthest from it's nearest cluster to be the next centroid
        for point in points:
            point_nearest_centroid = [] 
            for centroid in centroids:
                distance_squared = (centroid[0] - point[0])**2 + (centroid[1] - point[1])**2
                point_nearest_centroid.append(distance_squared)

            nearest_centroid = min(point_nearest_centroid)
            distance_from_nearest_point.append(nearest_centroid)

        centroid = points[distance_from_nearest_point.index(max(distance_from_nearest_point))]
        centroids.append(centroid)


def K_Means_Clustering(k, centroids, input_data):
    global clusters

    # Initializing clusters to hold the points allocated to k clusters
    clusters = []
    i = 0
    while i <= k - 1:
        i += 1
        clusters.append([])

    # Finding the nearest cluster for every point, and appending it to the cluster
    for point in input_data:
        distance_to_centroids = []
        for centroid in centroids:                                                                                                                                                                                                                      
            distance_squared = (centroid[0] - point[0])**2 + (centroid[1] - point[1])**2
            distance_to_centroids.append(distance_squared)

        nearest_cluster = clusters[distance_to_centroids.index(min(distance_to_centroids))]
        nearest_cluster.append(point)

    # Re-calculating centroids by finding the mean of all points in the cluster
    for cluster in range(0, len(clusters)):
        x_values = []
        y_values = []

        for point in clusters[cluster]:
            x = point[0]
            x_values.append(x)

            y = point[1]
            y_values.append(y)

            centroid = centroids[cluster]
            
        average_x = sum(x_values) / len(x_values)
        average_y = sum(y_values) / len(y_values)
            
        centroids[cluster] = [average_x, average_y]

'''Training K-Means + Results'''

def displaying_cluster(centroids, output_data):
    x_cent = []
    y_cent = []

    for centroid in centroids:  
        # Separating the x and y value of each centroid position   
        x_cent.append(centroid[0])
        y_cent.append(centroid[1])

    # Plotting the centroids on a scatter plot
    plt.scatter(x_cent, y_cent) 

    x_cent.clear()   
    y_cent.clear()

    # Repeating the process for all points(that aren't centroids) in the clusters
    for cluster in output_data:
        x_val = []
        y_val = []

        for point in cluster:
            x_val.append(point[0])
            y_val.append(point[1])

        plt.scatter(x_val, y_val)

    x_val.clear()
    y_val.clear()
    
    plt.savefig("clustered_output.png")
    plt.show()
    

def training_KMeans(k, input_data):
    # Centroids Initialization through K-Means++
    Centroid_Initialization(k, input_data)

    # Iteratively Running the K-Means Algorithm until the centroids of the clusters do not change for 2 consecutive iterations
    # 2 consecutive iterations lead to the best results in the least number of iterations, compared to a greater range like 3 or 5. 

    change_in_centroid = []
    changing = True
    while changing != False:
        K_Means_Clustering(k, centroids, points)

        # Recording the k centroids for all iterations until current iteration
        change_in_centroid.append([centroids])
        # For every iteration, select the last 2 elements of the change_in_centroid list to check for changes
        change_range = change_in_centroid[-2:]

        # Checking for changes in centroid positions
        for cent in change_range[:-1]:
            if cent == change_range[-1]:
                changing = False
            else:
                changing = True

def Most_Optimal_Number_Of_Cluster(inertial_values):
    most_optimal_number_of_clusters = []
    K = [5, 6, 7, 8, 9, 10]

    iters = 0
    while iters <= 1000:
        for k in K:
            # Performing clustering with k clusters
            training_KMeans(k, input_dataframe)

            # Finding the Inertia for this cluster configuration
            sum_of_distances_clusters = []
            for cluster in range(0, len(clusters)-1):
                within_cluster_distance = []
                for point in clusters[cluster]:
                    centroid = centroids[cluster]
                    euclidean_distance = (centroid[0] - point[0])**2 + (centroid[1] - point[1])**2
                    within_cluster_distance.append(euclidean_distance)
                
                within_cluster_sum_of_distance = sum(within_cluster_distance)
                sum_of_distances_clusters.append(within_cluster_sum_of_distance)
        
            inertia = sum(sum_of_distances_clusters)
            inertial_values.append(inertia) 
        
        # Finding the index maximum inertial value from all variations of k, to get the index of the most optimal number of clusters
        maximum_inertial_value_index = inertial_values.index(max(inertial_values))
        most_optimal_number_of_clusters.append(K[maximum_inertial_value_index])

        inertial_values.clear()
        
        iters += 1 

    num_of_clusters = max(set(most_optimal_number_of_clusters), key=most_optimal_number_of_clusters.count)
    print("The most optimal number of cluster is {}".format(num_of_clusters))

# Finding the most optimal number of clusters
all_inertial_values = []
Most_Optimal_Number_Of_Cluster(all_inertial_values)

# Training the K-Means Clustering model
training_KMeans(number_of_clusters, input_dataframe)

# Displaying Clustred Data Points:
displaying_cluster(centroids, clusters)
