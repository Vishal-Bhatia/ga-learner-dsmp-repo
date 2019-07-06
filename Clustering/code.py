# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

# Load Offers
##Loading the offers data
offers = pd.read_excel(path, sheet_name = 0)

# Load Transactions
##Loading the transactions data, and creating a new identification column for purchases
transactions = pd.read_excel(path, sheet_name = 1)
transactions["n"] = 1

# Merge dataframes
##Merging the two dataframes
df = offers.merge(transactions, on = offers.columns[0])

# Look at the first 5 rows
##Showing the top 5 rows
df.head()



# --------------
# Code starts here
# create pivot table
##Creating a pivot table as instructed
matrix = df.pivot_table(index = "Customer Last Name", columns = "Offer #", values = "n")

# replace missing values with 0
##Replacing the null values with zeroes
matrix.fillna(0, inplace = True)

# reindex pivot table
##Resetting the index as instructed
matrix.reset_index(inplace = True)

# display first 5 rows
##Showing the top 5 rows
matrix.head()

# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here
# initialize KMeans object
##Intantiating a KMeans clustering object
cluster = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)

# create 'cluster' column
##Plugging the cluster names into the primary dataframe [Note that we exclude the first column of the dataframe as that is a unique identifier]
matrix["cluster"] = cluster.fit_predict(matrix[matrix.columns[1:]])
matrix.head()

# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here
# initialize pca object with 2 components
##Intantiating a PCA object with 2 components
pca = PCA(n_components = 2, random_state = 0)

# create 'x' and 'y' columns donoting observation locations in decomposed form
##Creating the X-coordinate and y-coordinate dataframes
x = pca.fit_transform(matrix[matrix.columns[1:]])[:, 0]
y = pca.fit_transform(matrix[matrix.columns[1:]])[:, 1]
matrix["x"] = x
matrix["y"] = y

# dataframe to visualize clusters by customer names
##Creating clusters with specific columns as instructed
clusters = pd.DataFrame({matrix.columns[0]: matrix.iloc[:, 0], matrix.columns[33]: matrix.iloc[:, 33],matrix.columns[34]: matrix.iloc[:, 34], matrix.columns[35]: matrix.iloc[:, 35]})

# visualize clusters
##Plotting the clusters via a scatter plot
plt.scatter(x = clusters["x"], y = clusters["y"], c = clusters["cluster"], cmap = "viridis")

# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'clusters'
##Merging the clusters and clusters dataframes
data = pd.merge(clusters, transactions)

# merge `data` and `offers`
##Merging the data and offers dataframes
data = pd.merge(offers, data)

# initialzie empty dictionary
##INtantiating an empty dictionary for iteration over every cluster
champagne={}
# iterate over every cluster
for i in range(0,5):
    # observation falls in that cluster
    new_df = data[data["cluster"] == i]
    # sort cluster according to type of 'Varietal'
    counts = new_df["Varietal"].value_counts(ascending = False)
    # check if 'Champagne' is ordered mostly
    if counts.index[0] == "Champagne":
        # add it to 'champagne'
        champagne[i] = counts[0]

# get cluster with maximum orders of 'Champagne' 
##Outputting the cluster with maximum orders
cluster_champagne = max(champagne, key = champagne.get)
# print out cluster number
print(cluster_champagne)


# --------------
# Code starts here

# empty dictionary
##Initializing an empty dictionary for iteration over cluster counts
discount={}

# iterate over cluster numbers
for i in range(0, 5):
    # dataframe for every cluster
    new_df = data[data["cluster"] == i]
    # average discount for cluster
    counts = new_df["Discount (%)"].mean()
    # adding cluster number as key and average discount as value
    discount[i] = counts

# cluster with maximum average discount
##Obtaining teh cluster with the maximum discount
cluster_discount = max(discount, key = discount.get)

# Code ends here


