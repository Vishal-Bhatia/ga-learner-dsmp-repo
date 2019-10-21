# --------------
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix

# Code starts here
# load data
##Loading the CSV data onto a Pandas dataframe
news = pd.read_csv(path)

# subset data
##Subsetting the data as instructed
news = news[["TITLE", "CATEGORY"]]

# distribution of classes
# display class distribution
# display data
##Outputting the class distribution, and the first few obersvations
dist = news["CATEGORY"].value_counts()
print(dist)
print(news.head())

# Code ends here




# --------------
# Code starts here

# stopwords 
##Intantiating a stopwords object
stop = set(stopwords.words("english"))

# retain only alphabets
##Retaining only the alphabets in the "TITLE" column
news["TITLE"] = news["TITLE"].apply(lambda x: re.sub("[^a-zA-Z]", " ", x))

# convert to lowercase and tokenize
##Lower-casing and tokenizing the "TITLE" column
news["TITLE"] = news["TITLE"].apply(lambda x: (x.lower()).split())

# remove stopwords
##Removing stopwards from the "TITLE" column
news["TITLE"] = news["TITLE"].apply(lambda x: [word for word in x if word not in stop])

# join list elements
##Joining the elements in the "TITLE" column
news["TITLE"] = news["TITLE"].apply(lambda x: " ".join([word for word in x]))

# split into training and test sets
##Creating the train and test datasets
X_train, X_test, y_train, y_test = train_test_split(news["TITLE"], news["CATEGORY"], test_size = 0.2, random_state = 3)

# Code ends here




# --------------
# Code starts here

# initialize count vectorizer
# initialize tfidf vectorizer
##Intantiating a CountVectorizer object and a TF-IDF vectorizer object
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3))

# fit and transform with count vectorizer
# fit and transform with tfidf vectorizer
##Applying fit-transform on the training data, and transform on the testing data
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Code ends here




# --------------
# Code starts here

# initialize multinomial naive bayes
# fit on count vectorizer training data
# fit on tfidf vectorizer training data
# accuracy with count vectorizer
# accuracy with tfidf vectorizer
# display accuracies
##Intantiating two Multinomial Naive Bayes classifier models, one each for the count-vectorizer and TF-IDF data, fitting the same, and outputting the accuracy values
nb_1 = MultinomialNB()
nb_2 = MultinomialNB()
nb_1.fit(X_train_count, y_train)
nb_2.fit(X_train_tfidf, y_train)
acc_count_nb = accuracy_score(nb_1.predict(X_test_count), y_test)
acc_tfidf_nb = accuracy_score(nb_2.predict(X_test_tfidf), y_test)
print(acc_count_nb, acc_tfidf_nb)

# Code ends here




# --------------
import warnings
warnings.filterwarnings('ignore')

# initialize logistic regression
# fit on count vectorizer training data
# fit on tfidf vectorizer training data
# accuracy with count vectorizer
# accuracy with tfidf vectorizer
# display accuracies
##Intantiating two one-vs-rest Logistic Regression models, one each for the count-vectorizer and TF-IDF data, fitting the same, and outputting the accuracy values
logreg_1 = OneVsRestClassifier(LogisticRegression(random_state = 10))
logreg_2 = OneVsRestClassifier(LogisticRegression(random_state = 10))
logreg_1.fit(X_train_count, y_train)
logreg_2.fit(X_train_tfidf, y_train)
acc_count_logreg = accuracy_score(logreg_1.predict(X_test_count), y_test)
acc_tfidf_logreg = accuracy_score(logreg_2.predict(X_test_tfidf), y_test)
print(acc_count_logreg, acc_tfidf_logreg)


# Code ends here


