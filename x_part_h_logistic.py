"""
EE239AS Project 2
Author: Will, Hoooga, Xiaoyu, k
Part h
"""

from sklearn import svm, linear_model
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
lr = linear_model.LogisticRegression(C=100)
lr.fit(X_train, y_train)
# perform classification
y_predict = lr.predict(X_test)

# print metrics
print "Start logistic regression classifier"
print classification_report(y_test, y_predict)
print " The confusion matrix is\n", confusion_matrix(y_test,y_predict)       # print confusion matrix
print " Accuracy is " ,np.mean(y_predict == y_test)

y_score = lr.decision_function(X_test)
precison, recall, threshold = precision_recall_curve(y_test,y_score)
fpr, tpr, thr = roc_curve(y_test,y_score)
roc_auc = auc(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression: ROC Curve')
plt.plot(fpr,tpr)
plt.show()