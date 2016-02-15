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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string

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

hist = pd.DataFrame({'count': counts})

plt.figure()
hist.plot(kind='bar',alpha=0.5)

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
X_train = svd.transform(X_train_tfidf)
print X_train[0]
y_train = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in training_data['target']]

# train models
nb = GaussianNB().fit(X_train, y_train)
linear_svm = svm.SVC(kernel='linear',C=1).fit(X_train,y_train)

X_test_vec = vectorizer.transform(test_data['data'])
X_test_tfidf = tfidf.transform(X_test_vec)
X_test = svd.transform(X_test_tfidf)
y_test = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in test_data['target']]

y_predict = nb.predict(X_test)
print sum(y_predict == y_test) / float(len(y_predict))

y_score = linear_svm.decision_function(X_test)
precison, recall, threshold = precision_recall_curve(y_test,y_score)
plt.plot(precison,recall)

fpr, tpr, thr = roc_curve(y_test,y_score)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr)

# l = []
# for k in vectorizer.vocabulary_:
#     l.append((vectorizer.vocabulary_[k],k))
#
# l.sort(reverse=True)
