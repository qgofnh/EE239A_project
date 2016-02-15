from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems

vocab = ['The swimmer likes swimming so he swims.']
t = "The swimmer likes swimming."

vectorizer = CountVectorizer(tokenizer=tokenize,stop_words=text.ENGLISH_STOP_WORDS)
vectorizer.fit(vocab)
t_out = vectorizer.transform([t])

t_out = word_tokenize(t)

