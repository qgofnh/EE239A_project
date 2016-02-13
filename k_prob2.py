from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

stop_words = text.ENGLISH_STOP_WORDS

categories_computer = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
categories_recreation = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

com_train = fetch_20newsgroups(subset='train', categories=categories_computer, shuffle=True, random_state=42)
rec_train = fetch_20newsgroups(subset='train', categories=categories_recreation, shuffle=True, random_state=42)
com_test = fetch_20newsgroups(subset='test', categories=categories_computer, shuffle=True, random_state=42)
rec_test = fetch_20newsgroups(subset='test', categories=categories_recreation, shuffle=True, random_state=42)

categories_computer = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
categories_recreation = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']


count_vect = CountVectorizer()

X1_train_counts = count_vect.fit_transform(com_train.data)

#tf-idf
tfidf_transformer = TfidfTransformer()

X1_train_tfidf = tfidf_transformer.fit_transform(X1_train_counts)
print (X1_train_tfidf)

