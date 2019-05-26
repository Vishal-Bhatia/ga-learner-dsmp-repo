# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path

#Code starts here
##Loading the data onto a Pandas DataFrame
df = pd.read_csv(path)
print(df.head(5))

##Creating the X & y datasets
X = df.drop(["Price"], inplace = False, axis = 1)
y = df["Price"]

##Creating the training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)

##Finding the correlation among the predictor variables
corr = X_train.corr()
corr


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
##Intantiating the Linear Regression Model
regressor = LinearRegression()

##Fitting the above model and obtaining a prediction
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)




# --------------
from sklearn.linear_model import Lasso

# Code starts here
##Intantiating the Lasso Model
lasso = Lasso()

##Fitting the above model and obtaining a prediction
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
r2_lasso = r2_score(y_test, lasso_pred)




# --------------
from sklearn.linear_model import Ridge

# Code starts here
##Intantiating the Lasso Model
ridge = Ridge()

##Fitting the above model and obtaining a prediction
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
r2_ridge = r2_score(y_test, ridge_pred)



# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here
##Computing the cross-validation scores
score = cross_val_score(regressor, X_train, y_train, cv = 10)

##Computing the mean of the cross-validation scores
mean_score = np.mean(score)
mean_score


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here
##Initiating the pipeline for Polynomial Features
model = make_pipeline(PolynomialFeatures(2), LinearRegression())

##Fitting the model, obtaining predictions, and computing R-square
model.fit(X_train, y_train)
poly_pred = model.predict(X_test)
r2_poly = r2_score(y_test, poly_pred)




