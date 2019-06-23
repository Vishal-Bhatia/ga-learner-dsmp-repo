# --------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Code starts here
##Loading the CSV onto a Pandas dataframe and taking a look
df = pd.read_csv(path)
df.head(5)

##Outputting the first 5 columns of the dataframe
df[df.columns[0:5]]

##Creating the X and y dataframes
X = df.iloc[:, :-1]
y = df[df.columns[-1]]

##Creating the test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)

##Applying min-max scaling to standardize the data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Code ends here


# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

##Intantiating the Logistic Regression model, fitting the same, and computing the score 
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
roc_score = roc_auc_score(y_test, y_pred)



# --------------
from sklearn.tree import DecisionTreeClassifier

##Intantiating the Logistic Regression model, fitting the same, and computing the score 
dt = DecisionTreeClassifier(random_state = 4)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
roc_score = roc_auc_score(y_test, y_pred)




# --------------
from sklearn.ensemble import RandomForestClassifier

##Intantiating the Random Forest Classifier model, fitting the same, and computing the score 
rfc = RandomForestClassifier(random_state = 4)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
roc_score = roc_auc_score(y_test, y_pred)





# --------------
from sklearn.ensemble import BaggingClassifier

##Intantiating the Bagging Classifier model, fitting the same, and computing the score
bagging_clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(), n_estimators = 100, max_samples = 100, random_state = 0)
bagging_clf.fit(X_train, y_train)
score_bagging = bagging_clf.score(X_test, y_test)






# --------------
# Import libraries
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state = 4)
clf_3 = RandomForestClassifier(random_state = 4)

model_list = [("lr", clf_1), ("DT", clf_2), ("RF", clf_3)]

##Intantiating the Voting Classifier model, fitting the same, and computing the score
voting_clf_hard = VotingClassifier(estimators = model_list, voting = "hard")
voting_clf_hard.fit(X_train, y_train)
hard_voting_score = voting_clf_hard.score(X_test, y_test)





