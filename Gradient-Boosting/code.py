# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here
##Loading the CSV onto a Pandas dataframe and taking a look
df = pd.read_csv(path)
df.head(5)

##Creating the X and y dataframes
X = df.drop(["customerID", "Churn"], inplace = False, axis = 1)
y = df["Churn"]

##Creating the test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
##Replacing the spaces in the "TotalCharges" column in the test and train datasets
X_train["TotalCharges"] = X_train["TotalCharges"].replace(r'\s+', np.nan, regex = True)
X_test["TotalCharges"] = X_test["TotalCharges"].replace(r'\s+', np.nan, regex = True)

##Changing the dtype of the "TotalCharges" column in the test and train X datasets
X_train["TotalCharges"] = X_train["TotalCharges"].astype("float")
X_test["TotalCharges"] = X_test["TotalCharges"].astype("float")

##Filling the NaN values in the "TotalCharges" column in the test and train X datasets
X_train["TotalCharges"].fillna(np.mean(X_train["TotalCharges"]), inplace = True)
X_test["TotalCharges"].fillna(np.mean(X_train["TotalCharges"]), inplace = True)

##Checking if there are other columns with NaN values in the test and train X datasets
print(X_train.isnull().sum())
print(X_test.isnull().sum())

##Applying LabelEncoding on the test and train X datasets
label_encoder = LabelEncoder()
for i in X_train.columns:
    if X_train[i].dtype == "O":
        X_train[i] = label_encoder.fit_transform(X_train[i])
        X_test[i] = label_encoder.fit_transform(X_test[i])

##Replacing the "Yes" and "No" values in the test and train y datasets
binary = {"No": 0, "Yes": 1}
y_train = y_train.replace(binary, regex = True)
y_test = y_test.replace(binary, regex = True)


# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
##Outputting the X and y test and train datasets
print(X_train, X_test, y_train, y_test)

##Intantiating the AdaBoost model, fitting it, and making predictions
ada_model = AdaBoostClassifier(random_state = 0)
ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)

##Computing the accuracy score, the confusion matrix, and the classification report
ada_score = accuracy_score(y_test, y_pred)
ada_cm = confusion_matrix(y_test, y_pred)
ada_cr = classification_report(y_test, y_pred)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
##Intantiating the XGBoost model, fitting it, and making predictions
xgb_model = XGBClassifier(random_state = 0)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

##Computing the accuracy score, the confusion matrix, and the classification report
xgb_score = accuracy_score(y_test, y_pred)
xgb_cm = confusion_matrix(y_test, y_pred)
xgb_cr = classification_report(y_test, y_pred)

##Initializing the GridSearchCV object, fitting the same, and making predictions
clf_model = GridSearchCV(estimator = xgb_model, param_grid = parameters)
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)

##Computing the accuracy score, the confusion matrix, and the classification report
clf_score = accuracy_score(y_test, y_pred)
clf_cm = confusion_matrix(y_test, y_pred)
clf_cr = classification_report(y_test, y_pred)

##Outputting the two accuracy scores, confusion matrices, and classification reports
print(xgb_score, xgb_cm, xgb_cr)
print(clf_score, clf_cm, clf_cr)


