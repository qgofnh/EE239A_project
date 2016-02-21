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
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = word_tokenize(text)
    # tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems

categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']


training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

###############################################
############## initialize vectorizer, tfidf transformer and SVD
vectorizer = CountVectorizer(min_df=.01,stop_words=text.ENGLISH_STOP_WORDS,tokenizer=tokenize)
tfidf = TfidfTransformer(sublinear_tf=True, use_idf=True)
svd = TruncatedSVD(n_components=50, random_state=1, algorithm="arpack")

############## prepare training data
vectorizer.fit(training_data['data'])
print "finished fitting vectorizer"
X_train_vec = vectorizer.transform(training_data['data'])
print "finished vectorize raw data"
tfidf.fit(X_train_vec)
X_train_tfidf = tfidf.transform(X_train_vec)
print "finished tfidf transform"
svd.fit((X_train_tfidf))
X_train = svd.fit_transform(X_train_tfidf)
print "finished SVD dimension reduction"
y_train = training_data['target']

############## prepare testing data
X_test_vec = vectorizer.transform(test_data['data'])
X_test_tfidf = tfidf.transform(X_test_vec)
X_test = svd.transform(X_test_tfidf)
y_test = test_data['target']

############## train models
nb = MultinomialNB().fit(X_train_tfidf, y_train)
linear_svm_one_vs_one = svm.SVC(kernel='linear',C=1).fit(X_train_tfidf,y_train)
linear_svm_one_vs_rest = svm.LinearSVC().fit(X_train_tfidf,y_train)

############## perform classification
model = linear_svm_one_vs_rest

y_predict = model.predict(X_test_tfidf)
print model.score(X_test_tfidf,y_test)                  # mean accuracy

y_score = model.decision_function(X_test_tfidf)         # signed distance to the hyperplane
