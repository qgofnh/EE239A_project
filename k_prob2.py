from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

categories_computer = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
categories_recreation = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

com_train = fetch_20newsgroups(subset='train', categories=categories_computer, shuffle=True, random_state=42)
rec_train = fetch_20newsgroups(subset='train', categories=categories_recreation, shuffle=True, random_state=42)
com_test = fetch_20newsgroups(subset='test', categories=categories_computer, shuffle=True, random_state=42)
rec_test = fetch_20newsgroups(subset='test', categories=categories_recreation, shuffle=True, random_state=42)

categories_computer = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
categories_recreation = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

# tokenizing text

stop_words = text.ENGLISH_STOP_WORDS
# print(type(stop_words))
# count_vect = CountVectorizer('english')
count_vect = CountVectorizer(lowercase= True, max_df=0.95, stop_words = stop_words)


X1_train_counts = count_vect.fit_transform(com_train.data)

# tf-idf
tfidf_transformer = TfidfTransformer()
X1_train_tfidf = tfidf_transformer.fit_transform(X1_train_counts)
X1_array= X1_train_tfidf.toarray()
print('numbers of terms extracted: ', len(X1_array))








