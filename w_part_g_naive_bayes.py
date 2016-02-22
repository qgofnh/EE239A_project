"""
EE239AS Project 2
Author: Will, Hoooga, Xiaoyu, k

Part g
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
import cPickle,gzip
import numpy as np
import matplotlib.pyplot as plt

###############################################
#fetch tfidf_data
try:
    with gzip.open("tfidf_data.gz",'rb') as g:
      prepared_data=cPickle.load(g)
    g.close()
except:
    print "File not found!"

X_train = prepared_data['X_train_tfidf']
y_train = prepared_data['y_train']
X_test = prepared_data['X_test_tfidf']
y_test = prepared_data['y_test']

###############################################
nb = MultinomialNB().fit(X_train, y_train)          # model fitting
y_predict = nb.predict(X_test)                      # perform classification

# print metrics
target_names = ['rec', 'comp']
print classification_report(y_test, y_predict, target_names=target_names)   #classifcation report
print "The confusion matrix is\n", confusion_matrix(y_test,y_predict)       # print confusion matrix
print "The accuracy is: ", np.mean(y_predict == y_test)                     # accuracy

y_score = nb.predict_proba(X_test)                                          # ROC
fpr, tpr, thr = roc_curve(y_test,y_score[:,1])
roc_auc = auc(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes Classifier: ROC Curve')
plt.plot(fpr,tpr)
