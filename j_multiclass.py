from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsOneClassifier
import string
import numpy as np

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

############## initialize vectorizer, tfidf transformer and SVD
vectorizer = CountVectorizer(min_df=1,stop_words=text.ENGLISH_STOP_WORDS)
tfidf = TfidfTransformer(sublinear_tf=True, use_idf=True)
svd = TruncatedSVD(n_components=50, random_state=1, algorithm="arpack")

############## prepare training data
vectorizer.fit(training_data['data'])
X_train_vec = vectorizer.transform(training_data['data'])
tfidf.fit(X_train_vec)
X_train_tfidf = tfidf.transform(X_train_vec)
svd.fit((X_train_tfidf))
X_train = svd.fit_transform(X_train_tfidf)
y_train = training_data['target']
y_trainb = label_binarize(y_train, classes=[0, 1, 2, 3])
############## prepare testing data
X_test_vec = vectorizer.transform(test_data['data'])
X_test_tfidf = tfidf.transform(X_test_vec)
X_test = svd.transform(X_test_tfidf)
y_test = test_data['target']
y_testb = label_binarize(y_test, classes=[0, 1, 2, 3])

############## train models
models = [GaussianNB(), MultinomialNB(), OneVsOneClassifier(svm.SVC(kernel='linear',C=1)), svm.LinearSVC()]
model_names = ["GaussianNB","MultiNomialNB", "One-vs-One SVM", "One-vs-Rest SVM"]
for n in range(len(models)):
    print model_names[n]
    model = models[n]
    if n == 0 :
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_score = model.predict_proba(X_test)
    elif n==1:
       model.fit(X_train_tfidf, y_train)
       y_predict = model.predict(X_test_tfidf)
       y_score = model.predict_proba(X_test_tfidf)
    else:
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_score = model.decision_function(X_test) 
    # confusion matrix
    cm = confusion_matrix(y_test, y_predict)
    print "Confusion Matrix:"
    print cm
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print "Normalized Confusion Matrix"
    print cm_normalized
    # accuracy
    print 'accuracy = ', accuracy_score(y_test, y_predict)
    # precision recall
    print "Classification Report:"
    print(classification_report(y_test, y_predict,target_names=categories))
    # precision-recall curve
        # nothing
    # roc_curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure(n)
    color = ['r','g','b','c']
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_testb[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        #print 'roc_auc[',i,'] = ', roc_auc[i]        
        plt.plot(fpr[i], tpr[i], color[i], label='%s  (area = %0.2f)' % (categories[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of %s' %model_names[n])
    plt.legend(loc="lower right")  
plt.show()
