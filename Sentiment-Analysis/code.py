# --------------
##Importing the necessary libraries
import numpy as np
import pandas as pd

##Loading the CSV data on to a Pandas dataframe
df = pd.read_csv(path, sep = "\t")

##Checking the type of the date column, and making it a proper datetime series
df["date"].dtype
df["date"] = pd.to_datetime(df["date"])

##Creating a new column to capture the lenght of the reviews
df["length"] = df["verified_reviews"].apply(lambda row: len(row))




# --------------
##Importing the necessary classes
import matplotlib.pyplot as plt
import seaborn as sns

## Rating vs feedback
# set figure size
# generate countplot
# display plot
##Plotting countplot of rating vs feedback
sns.countplot(data = df, x = "rating", hue = "feedback")
plt.show()

## Product rating vs variation
# set figure size
# generate barplot
# display plot
##Plotting barplot of rating vs variation
sns.barplot(data = df, x = "rating", y = "variation", hue = "feedback")
plt.show()




# --------------
# import packages
##Importing the necessary libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# declare empty list 'corpus'
##Initializing an empty list to capture processed review data
corpus= []

##Intantiating a PorterStemmer object
ps = PorterStemmer()

# for loop to fill in corpus
    # retain alphabets
    # convert to lower case
    # tokenize
    # initialize stemmer object
    # perform stemming
    # join elements of list
    # add to 'corpus'
##Iterating over the reviews, processing them, and creating a corpus    
for i in range(0, 3150):
    review = re.sub("[^a-zA-Z]", " ", df["verified_reviews"][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review]
    review = [word for word in review if word not in stopwords.words("english")]
    review = " ".join(review)
    corpus.append(review)

# display 'corpus'
print(corpus)




# --------------
# import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Instantiate count vectorizer
# Independent variable
##Intantiating a CountVectorizer object, and fit-transforming the "headline_text" column to create the X dataset
cv = CountVectorizer(max_features = 1500)
cv.fit_transform(corpus)
vector = cv.fit_transform(corpus)
X = vector.toarray()

# dependent variable
##Creating the y dataset
y = df["feedback"]

# Counts
##Saving the value counts of the y dataset as a variable
count = df["feedback"].value_counts()

# Split the dataset
##Applying train-test split as instructed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)




# --------------
# import packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

# Instantiate calssifier
# fit model on training data
# predict on test data
# calculate the accuracy score
# calculate the precision
# display 'score' and 'precision'
##Intantiating a RandomForestClassifier model as instructed, fitting the same, and computing its accuracy
rf = RandomForestClassifier(random_state = 2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
score = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)




# --------------
# import packages
##Importing the necessary libraries
from imblearn.over_sampling import SMOTE

# Instantiate smote
# fit_sample onm training data
##Intantiating a SMOTE object and resampling the data
smote = SMOTE(random_state=9)
X_train, y_train = smote.fit_sample(X_train, y_train)

# fit model on training data
# predict on test data
# calculate the accuracy score
# calculate the precision
# display precision and score
##Fitting the earlier intantiated RandomForestClassifier model on resampled data and computing its accuracy
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
score = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print(score, precision)




