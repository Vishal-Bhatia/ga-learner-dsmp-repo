# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
##Loading the CSV data on to a Pandas dataframe and having a look
df = pd.read_csv(path)
print(df.head(5))
print(df.info())

##Removing the unnecessary strings from the monetary columns
for i in ["INCOME", "HOME_VAL", "BLUEBOOK", "OLDCLAIM", "CLM_AMT"]:
    df[i] = df[i].apply(lambda x: str(x).replace("$", "").replace(",", ""))

##Creating the X and y dataframes, and then splitting them
X = df.drop(["CLAIM_FLAG"], inplace = False, axis = 1)
y = df["CLAIM_FLAG"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)

# Code ends here


# --------------
# Code starts here
##Converting the monetary columns into float columns
for i in ["INCOME", "HOME_VAL", "BLUEBOOK", "OLDCLAIM", "CLM_AMT"]:
    X_train[i] = X_train[i].astype("float")
    X_test[i] = X_test[i].astype("float")

# Code ends here


# --------------
# Code starts here
##Removing the null rows from the X & y dataframes
X_train.drop(index = X_train[(X_train["YOJ"].isnull()) | (X_train["OCCUPATION"].isnull())].index,inplace = True)
X_test.drop(index = X_test[(X_test["YOJ"].isnull()) | (X_test["OCCUPATION"].isnull())].index, inplace = True)
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

##Replacing the null values with column means
X_train["AGE"].fillna((X_train["AGE"].mean()), inplace = True)
X_test["AGE"].fillna((X_train["AGE"].mean()), inplace = True)

X_train["CAR_AGE"].fillna((X_train["CAR_AGE"].mean()), inplace = True)
X_test["CAR_AGE"].fillna((X_train["CAR_AGE"].mean()), inplace = True)

X_train["INCOME"].fillna((X_train["INCOME"].mean()), inplace = True)
X_test["INCOME"].fillna((X_train["INCOME"].mean()), inplace = True)

X_train["HOME_VAL"].fillna((X_train["HOME_VAL"].mean()), inplace = True)
X_test["HOME_VAL"].fillna((X_train["HOME_VAL"].mean()), inplace = True)

# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1", "MSTATUS", "GENDER", "EDUCATION", "OCCUPATION", "CAR_USE", "CAR_TYPE", "RED_CAR", "REVOKED"]

# Code starts here
##Intantiating and implementing a Label Encoder
le = LabelEncoder()
for i in columns:
    X_train[i] = le.fit_transform(X_train[i])
    X_test[i] = le.transform(X_test[i])

# Code ends here


# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# code starts here 
##Intantiating a Logistic Regression model, fitting the same, and making predictions
model = LogisticRegression(random_state = 6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
##Intantiating SMOTE to balance the sample
smote = SMOTE(random_state = 9)
X_train, y_train = smote.fit_sample(X_train, y_train)

##Intantiating a standard scaler and applying the same
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Code ends here


# --------------
# Code Starts here
##Fitting the Logistic Regression model onto the resamples, scaled data
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
score

# Code ends here


