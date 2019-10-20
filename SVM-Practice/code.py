# --------------
import pandas as pd
from collections import Counter

# Load dataset
##Loading the CSV data onto a Pandas dataframe
data = pd.read_csv(path)

##Checking for null values
print(data.isnull().sum())

##Providing a statistical description of the above data
print(data.describe())


# --------------
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style = "darkgrid")

# Store the label values 
##Storing the target column as a variabl;e, and plotting the same
label = data["Activity"]
sns.countplot(x = label)
plt.xticks(rotation = 90)
plt.show()

# plot the countplot



# --------------
# make the copy of dataset
# Create an empty column 
##Creating a copy of the dataset, and then creating an empty column in the copy dataset
import numpy as np
data_copy = data.copy()
data_copy["duration"] = ""

# Calculate the duration
##Creating the groupby dataframe, & transforming the "duration" column
duration_df = (data_copy.groupby([label[label.isin(["WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"])], "subject"])["duration"].count() * 1.28)
duration_df = pd.DataFrame(duration_df)

# Sort the values of duration
##Creating the dataframe for plotting with the "duration" values sorted
plot_data = duration_df.reset_index().sort_values("duration", ascending = False)
plot_data["Activity"] = plot_data["Activity"].map({"WALKING_UPSTAIRS": "Upstairs", "WALKING_DOWNSTAIRS": "Downstairs"})

# Plot the durations for staircase use
##Plotting the barplot of "subject" vs "duration" for "Activity"
sns.barplot(data = plot_data, x = "subject", y = "duration", hue = "Activity")
plt.title("Participants Compared By Their Staircase Walking Duration")
plt.xlabel("Participants")
plt.ylabel("Total Duration [s]")
plt.show()



# --------------
#exclude the Activity column and the subject column
##Creating a sub-dataframe of only the continous variables
feature_cols = data.columns[: -2]

#Calculate the correlation values
#stack the data and convert to a dataframe
##Outputting the correlation matrix as an unstacked dataframe with column names as instructed
correlated_values = data[feature_cols].corr()
correlated_values = correlated_values.stack().to_frame().reset_index().rename(columns = {"level_0": "Feature_1", "level_1": "Feature_2", 0: "Correlation_score"})

#create an abs_correlation column
##Creating a column to capture the absolute correlation
correlated_values["abs_correlation"] = correlated_values["Correlation_score"].apply(lambda x: abs(x))

#Picking most correlated features without having self correlated pairs
##Creating a column to capture absolute correlation, and seggregating the highly-correlated pairs
top_corr_fields = correlated_values.sort_values("Correlation_score", ascending = False).query("abs_correlation > 0.8")
top_corr_fields = top_corr_fields[top_corr_fields["Feature_1"] != top_corr_fields["Feature_2"]].reset_index(drop=True)




# --------------
# importing neccessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import confusion_matrix, accuracy_score

# Encoding the target variable
##Intantiating and implementing a Label Encoder
le = LabelEncoder()
data["Activity"] = le.fit_transform(data["Activity"])

# split the dataset into train and test
##Creating the X and y dataframes, and performing the train-test split
X = data.drop("Activity", axis = 1)
y = data["Activity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 40)

# Baseline model 
##Intantiating a baseline SVC model
classifier = SVC(random_state = 40)

##Fitting the model
clf = classifier.fit(X_train, y_train)
y_pred = clf.predict(X_test)

##Outputting the precision, accuracy, and F1-score of the model, and storing the accuracy in a variable as suggested
precision, accuracy, f_score = error_metric(y_test, y_pred, average = "weighted")[0:3]
model1_score = accuracy_score(y_test, y_pred)



# --------------
# importing libraries
##Importing the necessary libraries
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

##Intantiating a LinearSVC model, and fitting the same
linsvc = LinearSVC(C = 0.01, penalty = "l1", dual = False, random_state = 42)
lsvc = linsvc.fit(X_train, y_train)

# Feature selection using Linear SVC
##Intantiating a "SelectFromModel" on our Linear SVC model, and creating new train and test features
model_2 = SelectFromModel(lsvc, prefit = True)
new_train_features = model_2.transform(X_train)
new_test_features = model_2.transform(X_test)

# model building on reduced set of features
##Intantiating another SVC model, fitting the same on the new features, and computing the error metrics
classifier_2 = SVC(random_state = 40)
clf_2 = classifier_2.fit(new_train_features, y_train)
y_pred_new = clf_2.predict(new_test_features)
precision, accuracy, f_score = error_metric(y_test, y_pred_new, average = "weighted")[0:3]
model2_score = accuracy_score(y_test, y_pred_new)




# --------------
# Importing Libraries
from sklearn.model_selection import GridSearchCV

# Set the hyperparmeters
# Usage of grid search to select the best hyperparmeters
##Intantiating a GridSearchCV object with the SVC as the primary model, and fitting the same on new features
parameters  = {"kernel": ["linear", "rbf"], "C": [100, 20, 1, 0.1]}
selector = GridSearchCV(estimator = SVC(), param_grid = parameters, scoring = "accuracy")
selector.fit(new_train_features, y_train)

##Outputting the best hyperparameters, and the detailed scores
print(selector.best_params_)
results = selector.cv_results_
means = results["mean_test_score"]
stds = results["std_test_score"]
params = results["params"]
for mean, std, params in zip(means, stds, params):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

# Model building after Hyperparameter tuning
##Intantiating a new SVC model with the best hyperparameter values, fitting the same, and outputting its error metrics
classifier_3 = SVC(C = 20, kernel = "rbf", random_state = 40)
clf_3 = classifier_3.fit(new_train_features, y_train)
y_pred_final = clf_3.predict(new_test_features)
precision, accuracy, f_score = error_metric(y_test, y_pred_final, average = "weighted")[0:3]
model3_score = accuracy_score(y_test, y_pred_final)




