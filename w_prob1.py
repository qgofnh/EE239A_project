from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
import numpy as np

# define functions to remove stemmed words
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

categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                        'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

counts = [sum(training_data['target'] == i) for i in range(8)]

############################################### a) plot histogram
# hist = pd.DataFrame({'count': counts})
# plt.figure()
# hist.plot(kind='bar',alpha=0.5)
###############################################
vectorizer = CountVectorizer(min_df=1,tokenizer=tokenize,stop_words=text.ENGLISH_STOP_WORDS)
tfidf = TfidfTransformer(sublinear_tf=True, use_idf=True)
svd = TruncatedSVD(n_components=50, random_state=42, algorithm="arpack")

vectorizer.fit(training_data['data'])
print "hello"
X_train_vec = vectorizer.transform(training_data['data'])
tfidf.fit(X_train_vec)
X_train_tfidf = tfidf.transform(X_train_vec)
svd.fit((X_train_tfidf))
############################################### b) the final number of terms extracted after TFxIDF is 65567
# print X_train_tfidf.shape
############################################### d) apply LSI to TFxIDF and map each document to a 50-dimensional vector
X_train = svd.transform(X_train_tfidf)
# print X_train[0]
y_train = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in training_data['target']]

# train models
nb = GaussianNB().fit(X_train, y_train)
# linear_svm = svm.SVC(kernel='linear',C=0).fit(X_train,y_train) # hard margin

################################################ f) 5 fold for soft margin svm
X_test_vec = vectorizer.transform(test_data['data'])
X_test_tfidf = tfidf.transform(X_test_vec)
X_test = svd.transform(X_test_tfidf)
y_test = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in test_data['target']]
kf = KFold(len(y_train), n_folds=5, shuffle = True)
y_train = np.asanyarray(y_train)
for train_index, test_index in kf:
    # print train_index
    X_train_k, X_test_k = X_train[train_index], X_train[test_index]
    y_train_k, y_test_k = y_train[train_index], y_train[test_index]
    linear_svm = svm.SVC(kernel='linear',C=0.1, gamma=0.001).fit(X_train_k,y_train_k) # soft margin, gamma from 10^-3 to 10^3
    y_predict_k = linear_svm.predict(X_test_k)
    # print metrics.classification_report(y_test_k,y_predict_k)
    # metrics.confusion_matrix(y_test_k,y_predict_k)
    print sum(y_predict_k == y_test_k) / float(len(y_predict_k))
################################################

# X_test_vec = vectorizer.transform(test_data['data'])
# X_test_tfidf = tfidf.transform(X_test_vec)
# X_test = svd.transform(X_test_tfidf)
# y_test = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in test_data['target']]

# y_predict = linear_svm.predict(X_test)
# print sum(y_predict == y_test) / float(len(y_predict))

# y_score = linear_svm.decision_function(X_test)
# precison, recall, threshold = precision_recall_curve(y_test,y_score)
# plt.plot(precison,recall)
#
# fpr, tpr, thr = roc_curve(y_test,y_score)
# roc_auc = auc(fpr,tpr)
# plt.plot(fpr,tpr)

# l = []
# for k in vectorizer.vocabulary_:
#     l.append((vectorizer.vocabulary_[k],k))
#
# l.sort(reverse=True)
