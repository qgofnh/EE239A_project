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
from nltk import WordPunctTokenizer
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
import cPickle,gzip

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # text = "".join([ch for ch in text if ch not in string.punctuation])

    tokens = word_tokenize(text)
    # tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems

def tokenize(txt):
    # txt = "".join([ch for ch in txt if ch not in string.punctuation])
    # tokens = word_tokenize(txt)
    # tokens = [i for i in tokens if i not in string.punctuation]

    tokens = re.findall('(?u)\\b\\w\\w+\\b',re.sub('[0-9.]*','',txt.lower()))
    tokens_wo_stop = [item for item in tokens if item not in text.ENGLISH_STOP_WORDS]
    stems = stem_tokens(tokens_wo_stop, stemmer)
    return stems

categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                        'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']


training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

counts = [sum(training_data['target'] == i) for i in range(8)]

hist = pd.DataFrame({'count': counts})

plt.figure()
hist.plot(kind='bar',alpha=0.5)

###############################################
# construct vectorizer, tfidf transformer and SVD
vectorizer = CountVectorizer(min_df=1,stop_words=stop_l,tokenizer=tokenize)
tfidf = TfidfTransformer(sublinear_tf=True, use_idf=True)
svd = TruncatedSVD(n_components=50, random_state=1, algorithm="arpack")

# prepare training data
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
y_train = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in training_data['target']]

# prepare testing data
X_test_vec = vectorizer.transform(test_data['data'])
X_test_tfidf = tfidf.transform(X_test_vec)
X_test = svd.transform(X_test_tfidf)
y_test = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in test_data['target']]

# train models
nb = MultinomialNB().fit(X_train, y_train)
linear_svm = svm.SVC(kernel='linear',C=1).fit(X_train_tfidf,y_train)

# perform classification
y_predict = linear_svm.predict(X_test_tfidf)
print sum(y_predict == y_test) / float(len(y_predict))

y_score = linear_svm.decision_function(X_test)
precison, recall, threshold = precision_recall_curve(y_test,y_score)
plt.plot(precison,recall)

fpr, tpr, thr = roc_curve(y_test,y_score)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr)

l = []
for k in vectorizer.vocabulary_:
    l.append((vectorizer.vocabulary_[k],k))

l.sort(reverse=True)

vocab = vectorizer.vocabulary_.keys()
vocab_new = [item for item in vocab if item == stemmer.stem(item)]
vectorizer = CountVectorizer(vocabulary=vocab_new)
vectorizer.fit_transform(test_data['data'])

analyzer = vectorizer.build_analyzer()
token = vectorizer.build_tokenizer()
pre = vectorizer.build_preprocessor()