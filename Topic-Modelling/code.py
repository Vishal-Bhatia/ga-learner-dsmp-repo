# --------------
# import libraries
import numpy as np
import pandas as pd
import re

# Load data
##Loading the CSV data onto a Pandas dataframe, and then subsetting as suggested
data = pd.read_csv(path, parse_dates = [0], infer_datetime_format = True)

# Sort headlines by date of publish
##Sorting the data as per publishing date
data.sort_values("publish_date", inplace = True)

# Retain only alphabets
##Retaining only the alphabets
data["headline_text"] = data["headline_text"].apply(lambda x: re.sub("[^a-zA-Z]", " ", x))

# Look at the shape of data
##Checking the shape of the data
data.shape

# Look at the first first five observations
##Checking the top 5 rows ofthe data
data.head()




# --------------
# import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
# Transform headlines
##Intantiating a CountVectorizer object, and fit-transforming the "headline_text" column
vectorizer = CountVectorizer(stop_words = "english", max_features = 30000)
news = vectorizer.fit_transform(data["headline_text"])

# initialize empty dictionary
# initialize with 0
##Initializing an empty dictionary  and i as 0
words = {}
i = 0

# Number of time every feature appears over the entire document
##Calculating the number of times every word/feature appears in the corpus
sums = np.array(np.sum(news, axis=0)).flatten()

# Loop to map 'sums' to its word
##Creating a for loop to populate the above dictionary
for word in vectorizer.get_feature_names():
    words[word] = sums[i]
    i += 1
    
# Top 20 most occuring words
##Seggregating the 20 most occuring words, and capturing tehse and their values
top_20 = sorted(words.items(), key = operator.itemgetter(1), reverse = True)[:20]
top_20_words = [i[0] for i in top_20]
top_20_values = [i[1] for i in top_20]

# Display top 20 words
##Plotting the 20 most occuring words
sns.barplot(x = top_20_words, y = top_20_values)
plt.title("Top 20 Words")
plt.xlabel("Words")
plt.ylabel("Values")
plt.show()




# --------------
# import libraries
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import pprint

# number of topics
##Setting a number for topic count
n_topics = 5

# initialize SVD
# fit and transform 'news'
##Intantiating a truncated SVD model, and fit-transforming the vectorized data
lsa_model = TruncatedSVD(n_components = n_topics, random_state = 2)
lsa_topic_matrix = lsa_model.fit_transform(news)


'''We are not interested in knowing every word of a topic.
Instead, we want to look at the first (lets say) 10 words
of a topic'''


# empty dictionary to store topic number and top 10 words for every topic 
# loop over every topic
##Initializing an empty dictionary to iteratively capture topic number and top 10 words for every topic
topic_lsa = {}
for i, topic in enumerate(lsa_model.components_):
    key = "Topic {}".format(i)
    value = [(vectorizer.get_feature_names()[i] + "*" + str(topic[i])) for i in topic.argsort()[:-11:-1]]
    topic_lsa[key] = " + ".join(value)
    
# pretty print topics
##Outputting the topic dictionary
pprint.pprint(topic_lsa)




# --------------
# import libraries
import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from nltk import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import matplotlib.pyplot as plt

# Function to clean data from stopwords, punctuation marks and lemmatize
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


# Code starts here

# stopwords list
##Intantiating a stopwords object
stop = set(stopwords.words("english"))

# string punctuations
##Setting up a punctuation excluder
exclude = set(string.punctuation)

# lemmatizer
##Intantiating a word-lemmatizer object
lemma = WordNetLemmatizer()

# convert headlines to list
##Outputting the "headlines_text" column as a list
headlines = data["headline_text"].tolist()

# cleaned data
##Cleaning/pre-processing the headlines data
clean_headlines = [clean(doc).split() for doc in headlines]

# Creating the term dictionary of our courpus, where every unique term is assigned an index
##Creating the id2word dictionary
dictionary = corpora.Dictionary(clean_headlines)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
##Creating a word corpus
doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_headlines]

# build LDA model
# extract topics for headlines
# pprint topics
##Intantiating the LDA model, and printing the 5 topics containing only the 10 most relevant words
lda_model = LdaModel(corpus = doc_term_matrix, num_topics = 5, id2word = dictionary, iterations = 10, random_state = 2)
topics = lda_model.print_topics(num_topics = 5, num_words = 10)
pprint.pprint(topics)

# Code ends here




# --------------
# Can take a long time to run
# coherence score
##Intantiating a coherence object, and outputting the coherence score
coherence_model_lda = CoherenceModel(model = lda_model, texts = clean_headlines, dictionary = dictionary, coherence = "c_v")
coherence_lda = coherence_model_lda.get_coherence()
print(coherence_lda)

# Function to calculate coherence values
##Initializing two empty lists, and defining a function to capture coherence values by number of topics
coherence_values = []
model_list = []
def compute_coherence_values(dictionary, corpus, texts, limit, start = 2, step = 3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    for num_topics in range(2, 50, 6):
        Lda = gensim.models.ldamodel.LdaModel
        model = Lda(doc_term_matrix, num_topics = num_topics, random_state = 2, id2word = dictionary, iterations = 10)
        model_list.append(model)
        coherencemodel = CoherenceModel(model = model, texts = texts, dictionary = dictionary, coherence = "c_v")
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary = dictionary, corpus = doc_term_matrix, texts = clean_headlines, start = 2, limit = 50, step = 6)

# Plotting
##Plotting coherence values against the number of topics
x = range(2, 50, 6)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.show()


