# Implementing K-Means Clustering from Scratch

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the Mall Customers CSV File
df = pd.read_csv("Mall_Customers.csv")
print(df.to_string())

'''Creating K-Means Clustering'''

number_of_clusters = 