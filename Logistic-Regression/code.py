# --------------
#Task 1 - Data loading and splitting

# import the libraries
##Importing the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Code starts here
##Loading the file onto a Pandas dataframe and running a check
df = pd.read_csv(path)
print(df.head())

##Ceating the X and y dataframes
X = df.drop(columns = "insuranceclaim")
y = df["insuranceclaim"]

##Splitting the X and y into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 6)

# Code ends here


# --------------
#Task 2 - Outlier  Detection
##Importing the necessary library
import matplotlib.pyplot as plt

# Code starts here
##Plotting the boxplot as instructed
plt.boxplot(X_train["bmi"])

##Saving the quantile as instructed
q_value = X_train["bmi"].quantile(0.95)

##Outputting the value counts as instructed
print(y_train.value_counts())

# Code ends here


# --------------
#Task 3 - Correlation Check !
# Code starts here
##Saving the correlation matrix as a variable, and outputting the same
relation = X_train.corr()
print(relation)

#Plotting the figure as instructed
plt.figure(figsize = (10, 8))
sns.pairplot(X_train)
plt.show()

# Code ends here


# --------------
#Task 4 - Predictor check!
##Importing the necessary packages
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here
##Saving the column names in a variable
cols = ["children", "sex", "region", "smoker"]

##Setting the subplot
fig, axes = plt.subplots(nrows = 2, ncols = 2,  figsize = (20, 20))

##Iterating over the subplots using nested-for loops
for i in range(0, 2):
   for j in range(0, 2):
       col = cols[i*2 + j]
       axes[i, j].set_title(cols)
       sns.countplot(x = X_train[col], hue = y_train, ax = axes[i,j])
       axes[i, j].set_xlabel(col)
       axes[i, j].set_ylabel("Insurance Claim")

# Code ends here


# --------------
#Task 5 - Claim prediction!

##Impoting the necessary modules
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

##Setting the parameters for grid search
parameters = {"C": [0.1, 0.5, 1, 5]}

# Code starts here
##Intantiating the logistic regression method, and identifying the correct parameters
lr = LogisticRegression()
grid = GridSearchCV(estimator=lr, param_grid = parameters)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Code ends here


# --------------
##Importing the necessary classes
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here
##Computing the AU-ROC score and plotting the AU-ROC curve
score = roc_auc_score(y_test, y_pred)
y_pred_proba = grid.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = "Logistic model, auc=" + str(roc_auc))
print(roc_auc)

# Code ends here


