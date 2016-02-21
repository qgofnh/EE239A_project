"""
EE239AS Project 2
Author: Will, Hoooga, Xiaoyu, k
Part f
"""

from sklearn import svm, metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cross_validation import KFold
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
y_train = np.asanyarray(y_train)
X_test = prepared_data['X_test']
y_test = prepared_data['y_test']

###############################################
print "Start 5-fold cross validation for soft margin svm"
kf = KFold(len(y_train), n_folds=5, shuffle = True)
for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    accuracy = [];
    for train_index, test_index in kf:
        X_train_k, X_test_k = X_train[train_index], X_train[test_index]
        y_train_k, y_test_k = y_train[train_index], y_train[test_index]
        # train models
        linear_svm = svm.SVC(kernel='linear',C=C).fit(X_train_k,y_train_k)          # svc with C from 0.001 to 1000 is a soft margin
        y_predict_k = linear_svm.predict(X_test_k)
        accuracy.append(np.mean(y_predict_k == y_test_k))
    print ' When C is ', C, ', average accuracy of 5 fold cross validation is ', np.mean(accuracy)

print "Start soft margin svm for all training data when C is 10"
linear_svm = svm.SVC(kernel='linear',C=10).fit(X_train,y_train)
y_predict = linear_svm.predict(X_test)
print metrics.classification_report(y_test,y_predict)
print " The confusion matrix is\n", confusion_matrix(y_test,y_predict) # print confusion matrix
print ' Accuracy is ', np.mean(y_predict == y_test)

y_score = linear_svm.decision_function(X_test) # print ROC curve
precison, recall, threshold = precision_recall_curve(y_test,y_score)
fpr, tpr, thr = roc_curve(y_test,y_score)
roc_auc = auc(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Soft Margin SVM: ROC Curve')
plt.plot(fpr,tpr)
plt.show()


