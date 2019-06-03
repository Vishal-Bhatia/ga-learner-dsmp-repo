# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here
# read the dataset
##Loading the data on to a Pandas dataframe
dataset = pd.read_csv(path)

# look at the first five columns
##Checking the first 5 rows
dataset.head(5)

# Check if there's any column which is not useful and remove it like the column id
##Dropping the Id column
dataset.drop("Id", inplace = True, axis = 1)

# check the statistical description
dataset.describe()


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
##Saving the column names as instructed
cols = dataset.columns

#number of attributes (exclude target)
size = len(cols) - 1

#x-axis has target attribute to distinguish between classes
#y-axis shows values of an attribute
##Creating the reverse dependent and independent variables
x = dataset["Cover_Type"]
y = dataset.drop(["Cover_Type"], inplace = False, axis = 1)

#Plot violin for all attributes
##Plotting the violin plots
for i in range(size):
    sns.violinplot(y = y[cols[i]])
    plt.show()




# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5

# Code Starts Here
##Subsetting the dataset with only the first 10 columns
subset_train = y[cols[0:10]]

##Computing the Pearson's correlation coefficient matrix
data_corr = subset_train.corr()

##Plotting a heatmap of the Pearson's correlation coefficient matri
sns.heatmap(data_corr, annot = True)
plt.show()

##Storing the correlation values in a variable
correlation = data_corr.unstack().sort_values(kind = "quicksort")
corr_var_list = correlation[(correlation != 1) & ((correlation > upper_threshold) | (correlation < lower_threshold))]

# Code ends here




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it 
##Dropping the unnecessary columns
dataset.drop(columns = ["Soil_Type7", "Soil_Type15"], inplace = True)

##Splitting the data for cross-validation
X = dataset.drop(["Cover_Type"], inplace = False, axis = 1)
Y = dataset["Cover_Type"]
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
#Standardized
##Intantiating the Standard Scaler
scaler = StandardScaler()

#Apply transform only for non-categorical data
##Applying the scaler on the non-categorical data
X_train_temp = scaler.fit_transform(X_train.iloc[:, :10])
X_test_temp = scaler.transform(X_test.iloc[:, :10])

#Concatenate non-categorical data and categorical
##Concatenating the scaled non-categorical and categorical data
X_train1 = numpy.concatenate((X_train_temp, X_train.iloc[:, 10:len(dataset.columns) - 1]), axis = 1)
X_test1 = numpy.concatenate((X_test_temp, X_test.iloc[:,10:len(dataset.columns) - 1]), axis = 1)

scaled_features_train_df = pd.DataFrame(X_train1, index = X_train.index, columns = X_train.columns)
scaled_features_test_df = pd.DataFrame(X_test1, index = X_test.index, columns = X_test.columns)


# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:
##Intantiating the SelectPercentile feature selection model
skb = SelectPercentile(score_func = f_classif, percentile = 20)

##Predicting using the above model and computing the scores
predictors = skb.fit_transform(X_train1, Y_train)
scores = skb.scores_

##Creating a list containing the top 10 column names and their scores
Features = X_train.columns
top_k_predictors=list(pd.DataFrame({"Features": Features, "Scores": scores}).sort_values(by = "Scores",ascending = False).iloc[:11, 0])


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

##Intantiating the classifier models as instructed
clf = OneVsRestClassifier(estimator = LogisticRegression())
clf1 = OneVsRestClassifier(estimator = LogisticRegression())

##Computing values for the all-features model
model_fit_all_features = clf1.fit(X_train, Y_train)
predictions_all_features = model_fit_all_features.predict(X_test)
score_all_features = accuracy_score(Y_test, predictions_all_features)

##Computing values for the top-features model
model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors], Y_train)
predictions_top_features = model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])
score_top_features = accuracy_score(Y_test, predictions_top_features)


