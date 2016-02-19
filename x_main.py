# Proj2 a)
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, linear_model
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import numpy as np
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

categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
# fetch training data and test data
training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
counts = [sum(training_data['target'] == i) for i in range(8)]

#################################### a) plot histogram
# plt.bar(np.arange(len(counts)), counts, 0.9, align='center', alpha=0.5)
# plt.xticks(np.arange(len(counts)),categories,rotation = 20,ha = 'right')
# plt.xlabel('Categories')
# plt.ylabel('Count')
# plt.show()
#################################### f) SVM Algorithm
text_clf = Pipeline([('vect', CountVectorizer(min_df=1,tokenizer=tokenize,stop_words=text.ENGLISH_STOP_WORDS)),
                     ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True)),
                     ('svd', TruncatedSVD(n_components=50, random_state=42, algorithm="arpack"))
])
text_clf.fit(training_data.data)
X_train = text_clf.transform(training_data.data)
text_clf.fit(test_data.data)
X_test = text_clf.transform(test_data.data)

y_train = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in training_data['target']]
y_test = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in test_data['target']]
kf = KFold(len(y_train), n_folds=5, shuffle = True)
y_train = np.asanyarray(y_train)
for train_index, test_index in kf:
    # print train_index
    X_train_k, X_test_k = X_train[train_index], X_train[test_index]
    y_train_k, y_test_k = y_train[train_index], y_train[test_index]
    linear_svm = svm.SVC(kernel='linear',C=0.1, gamma=0.001).fit(X_train_k,y_train_k) # soft margin, C is 0.1, gamma from 10^-3 to 10^3
    y_predict_k = linear_svm.predict(X_test_k)
    # print metrics.classification_report(y_test_k,y_predict_k) # print confusion matrix
    # metrics.confusion_matrix(y_test_k,y_predict_k)
    print sum(y_predict_k == y_test_k) / float(len(y_predict_k))

#################################### g) Gaussian Naive Bayes Classifier
text_clf = Pipeline([('vect', CountVectorizer(min_df=1,tokenizer=tokenize,stop_words=text.ENGLISH_STOP_WORDS)),
                     ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True)),
                     ('svd', TruncatedSVD(n_components=50, random_state=42, algorithm="arpack")),
                     ('clf', GaussianNB())
])
y_train = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in training_data['target']]
nb = text_clf.fit(training_data.data, y_train)
y_predict = nb.predict(test_data.data)
y_test = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in test_data['target']]
print sum(y_predict == y_test) / float(len(y_predict))
#################################### h) Logistic Regression
text_clf = Pipeline([('vect', CountVectorizer(min_df=1,tokenizer=tokenize,stop_words=text.ENGLISH_STOP_WORDS)),
                     ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True)),
                     ('svd', TruncatedSVD(n_components=50, random_state=42, algorithm="arpack")),
                     ('lr', linear_model.LogisticRegression(C=1e5))
])
y_train = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in training_data['target']]
nb = text_clf.fit(training_data.data, y_train)
y_predict = nb.predict(test_data.data)
y_test = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in test_data['target']]
print sum(y_predict == y_test) / float(len(y_predict))
####################################

print 'end'



