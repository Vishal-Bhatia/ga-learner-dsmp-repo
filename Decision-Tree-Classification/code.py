# --------------
#Importing header files
import pandas as pd
from sklearn.model_selection import train_test_split

# Code starts here
##Loading the CSV onto a Pandas dataframe
data = pd.read_csv(path)

##Creating the X & y dataframes and then splitting them into training and testing datasets
X = data.drop(["customer.id", "paid.back.loan"], inplace = False, axis = 1)
y = data["paid.back.loan"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
##Saving the value counts of "paid.back.loan" as instructed
fully_paid = y_train.value_counts()

##Plotting the bar graph
plt.bar(x = fully_paid.index, height = fully_paid)

# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
##Reformatting the "int.rate" column as instructed
X_train["int.rate"] = X_train["int.rate"].str.rstrip("%").astype("float")/100
X_test["int.rate"] = X_test["int.rate"].str.rstrip("%").astype("float")/100

##Seggregating the numeric and object column types
num_df = X_train.select_dtypes(include = ["int", "float"])
cat_df = X_train.select_dtypes(include = "object")

# Code ends here


# --------------
#Importing header files
import seaborn as sns

# Code starts here
##Saving the names of numerical columns as a list
cols = num_df.columns

##Setting the subplot
fig, axes = plt.subplots(nrows = 9, ncols = 1,  figsize = (20, 20))

##Iterating over the subplots using a for loop
for i in range(0, 9):
    axes[i].set_title("Paid Back Loan vs. " + cols[i])
    sns.boxplot(x = y_train, y = num_df[cols[i]], ax = axes[i])
    axes[i].set_xlabel("Paid Back Loan")
    axes[i].set_ylabel(cols[i])

# Code ends here


# --------------
# Code starts here
##Saving the names of object columns as a list
cols = cat_df.columns

##Setting the subplot
fig, axes = plt.subplots(nrows = 2, ncols = 2,  figsize = (20, 20))

##Iterating over the subplots using nested-for loops
for i in range(0, 2):
   for j in range(0, 2):
       axes[i, j].set_title(cols[i*2 + j])
       sns.countplot(x = X_train[cols[i*2 + j]], hue = y_train, ax = axes[i, j])
       axes[i, j].set_xlabel(cols[i*2 + j])
       axes[i, j].set_ylabel("Paid Back Loan")

# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier

# Code starts here
##Intantiating a LabelEnoder
le = LabelEncoder()

##Iterating over the categorical columns of the X dataframes
for i in cat_df.columns:
    X_train[i].fillna("NA", inplace = True)
    X_train[i]= le.fit_transform(X_train[i])
    X_test[i].fillna("NA", inplace = True)
    X_test[i]= le.fit_transform(X_test[i])

##Replacing the "No" and "Yes" values in the y dataframes with 0 & 1
y_train.map(dict(Yes = 1, No = 0))
y_test.map(dict(Yes = 1, No = 0))

##Intantiating the DecisionTree model, fitting the same, and computing the accuracy score on the testing dataframes
model = DecisionTreeClassifier(random_state = 0)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3, 10), 'min_samples_leaf': range(10, 50, 10)}

# Code starts here
##Intantiating another DecisionTree model along with grid parameters, fitting the same, and computing the accuracy score on the testing dataframes
model_2 = DecisionTreeClassifier(random_state = 0)
p_tree = GridSearchCV(estimator = model_2, param_grid = parameter_grid, cv = 5)
p_tree.fit(X_train, y_train)
acc_2 = p_tree.score(X_test, y_test)

# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
##Plotting the tree as instructed
dot_data = export_graphviz(decision_tree = p_tree.best_estimator_, out_file = None, feature_names = X.columns, filled = True, class_names = ["loan_paid_back_yes", "loan_paid_back_no"])
graph_big = pydotplus.graph_from_dot_data(dot_data)

# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


