# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
##Loading the file ontop a pandas dataframe
df = pd.read_csv(path)
print(df.head(5))

##Creating the dependent (X) and independent (y) variables
X = df.drop(["list_price"], inplace = False, axis = 1)
y = df["list_price"]

##Creating the testing and training datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)

# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        
##Creating the list of columns of X
cols = X_train.columns

##Plotting the figure
k = 0
fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (5, 5))
for i in [1, 2, 3]:
    row = i
    i = i + 1
    for j in [1, 2, 3]:
        axes.flatten()[k].scatter(x = X_train[cols[k]], y = y_train)
        j = j + 1
        k = k + 1

# code ends here


# --------------
# Code starts here
##Creating the correlation matrix variable and outputting the same
corr = X_train.corr(method = "pearson")
print(corr)

##Dropping the correlated dependent variables
X_train.drop(["play_star_rating", "val_star_rating"], inplace = True, axis = 1)
X_test.drop(["play_star_rating", "val_star_rating"], inplace = True, axis = 1)

# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here
##Initiating the linear regression
regressor = LinearRegression()

##Fitting the model on the testing and training data
regressor.fit(X_train, y_train)

##Making predictions
y_pred = regressor.predict(X_test)

##Computing the measn-squared error
mse = mean_squared_error(y_test, y_pred)
print(mse)

##Computing the r-square
r2 = r2_score(y_test, y_pred) 
print(r2)

# Code ends here


# --------------
# Code starts here
##Computing the residual
residual = y_test - y_pred

##Plotting a histogram of the residuals
residual.plot.hist(bins = 100)


# Code ends here


