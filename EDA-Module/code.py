# --------------
#Importing header files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Code starts here
##Loading the dataset onto a Pandas DataFrame
data = pd.read_csv(path)

##Plotting a histogram of "Rating"
data["Rating"].hist()

##Subsetting the dataframe as instructed
data = data[data["Rating"] <= 5]

##PLotting the histogram of "Rating"
plt.hist(data['Rating'])

#Code ends here


# --------------
# code starts here
##Assigning variables capturing all null value counts and frequencies
total_null = data.isnull().sum()
percent_null = 100*(total_null/data.isnull().count())

##Concatenating the two variables, and printing out the same
missing_data = pd.concat([total_null, percent_null], keys = ["Total", "Percent"], axis = 1, sort = False)
print(missing_data)

##Dropping rows with NaN values
data = data.dropna(axis = 0)

##Assigning variables capturing all null value counts and frequencies in the cut dataframe
total_null_1 = data.isnull().sum()
percent_null_1 = 100*(total_null/data.isnull().count())

##Concatenating the two variables, and printing out the same
missing_data_1 = pd.concat([total_null_1, percent_null_1], keys = ["Total", "Percent"], axis = 1, sort = False)
print(missing_data_1)

# code ends here


# --------------

#Code starts here
##PLotting as instructed
sns.catplot(x = "Category", y = "Rating", data = data, kind = "box", height = 10)
plt.xticks(rotation = 90)
plt.title("Rating vs Category [BoxPlot]") 

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
##Checkingthe value-counts of "Installs"
data["Installs"].value_counts()

##Replacing the commas and plus signs from "Installs"
data["Installs"] = data["Installs"].replace(regex = True, inplace = False, to_replace = r'\D', value = r'')
data["Installs"] = data["Installs"].apply(lambda x: int(x))

##Using "LabelEncoder" to transform the "Installs" column, and plotting the same
le = LabelEncoder()
le.fit(data["Installs"])
data["Installs"] = le.transform(data["Installs"])
sns.regplot(x = "Installs", y = "Rating", data = data)

#Code ends here



# --------------
#Code starts here
##Checkingthe value-counts of "Installs"
print(data["Price"].value_counts())

##Replacing the dollar signs from "Price"
data["Price"] = data["Price"].replace(regex = True, inplace = False, to_replace = r'\D', value = r'')
data["Price"] = data["Price"].apply(lambda x: float(x))

##Using "LabelEncoder" to transform the "Installs" column, and plotting the same
sns.regplot(x = "Price", y = "Rating", data = data)
plt.title("Rating vs Price [RegPlot]") 

#Code ends here


# --------------

#Code starts here
##Checking the unique values of the "Genres"column
data["Genres"].unique()

##Ensuring that the "Genres" column contains only the first value
data["Genres"] = data["Genres"].apply(lambda x: x if int(x.find(';')) == -1 else x[0: int(x.find(';'))])

##Using groupby to obtain mean ratings by genre, describing the data, and and sorting as instructed
gr_mean = data.groupby("Genres", as_index = False)["Rating"].mean()
print(gr_mean.describe())
gr_mean.sort_values("Rating", ascending = True, inplace = True)

##Outputting teh top- and bottom-most values
print(gr_mean.head(1))
print(gr_mean.tail(1))

#Code ends here


# --------------

#Code starts here
##Describing and visualizing the column "Last Updated"
print(data["Last Updated"].describe())
sns.distplot(data["Last Updated"].value_counts())

##Converting the "Last Updated" column to a proper datetime column
data["Last Updated"] = pd.to_datetime(data["Last Updated"])

##Obtaining the latest/max date and creating a new column that captures the gap from this date
max_date = data["Last Updated"].max()
data["Last Updated Days"] = (max_date - data["Last Updated"]).dt.days

##Plotting the new column
sns.regplot(x = "Last Updated Days", y = "Rating", data = data)
plt.title("Rating vs Last Updated [RegPlot]")

#Code ends here


