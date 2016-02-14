from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

categories_computer = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
categories_recreation = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
categories_total = categories_computer+categories_recreation


data_train = fetch_20newsgroups(subset='train', categories=categories_total, shuffle=True, random_state=42)
# tokenizing text

stop_words = text.ENGLISH_STOP_WORDS

count_vect = CountVectorizer(lowercase= True, max_df=0.95, stop_words= stop_words)


X1_train_counts = count_vect.fit_transform(data_train.data)
# test
# a= count_vect.vocabulary_.get('document')
# print(a)

# # tf-idf
tfidf_transformer = TfidfTransformer()
X1_train_tfidf = tfidf_transformer.fit_transform(X1_train_counts)
X1_array= X1_train_tfidf.toarray()
print('numbers of terms extracted: ', len(X1_array))









