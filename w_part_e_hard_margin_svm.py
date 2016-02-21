"""
EE239AS Project 2
Author: Will, Hoooga, Xiaoyu, k

Part e
"""

from sklearn import svm
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
    with gzip.open("prepared_data.gz",'rb') as g:
      prepared_data=cPickle.load(g)
    g.close()
except:
    print "File not found!"

X_train = prepared_data['X_train']
y_train = prepared_data['y_train']
X_test = prepared_data['X_test']
y_test = prepared_data['y_test']

###############################################
# train models
linear_svm = svm.SVC(kernel='linear',C=10000).fit(X_train,y_train)          # svc with a large enough C is essentially hard margin

# perform classification
y_predict = linear_svm.predict(X_test)

# print metrics
print classification_report(y_test, y_predict)
print "The confusion matrix is\n", confusion_matrix(y_test,y_predict)       # print confusion matrix
print np.mean(y_predict == y_test)

y_score = linear_svm.decision_function(X_test)
precison, recall, threshold = precision_recall_curve(y_test,y_score)
# plt.plot(precison,recall)
#
fpr, tpr, thr = roc_curve(y_test,y_score)
roc_auc = auc(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Hard Margin SVM: ROC Curve')
plt.plot(fpr,tpr)
