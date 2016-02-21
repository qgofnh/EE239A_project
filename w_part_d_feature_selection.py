"""
EE239AS Project 2
Author: Will, Hoooga, Xiaoyu, k

Part d
"""

import numpy as np
import cPickle,gzip
from sklearn.decomposition import TruncatedSVD

###############################################
#fetch tfidf_data
try:
    with gzip.open("tfidf_data.gz",'rb') as g:
      tfidf_data=cPickle.load(g)
    g.close()
except:
    print "File not found!"

X_train_tfidf = tfidf_data['X_train_tfidf']
y_train = tfidf_data['y_train']
X_test_tfidf = tfidf_data['X_test_tfidf']
y_test = tfidf_data['y_test']

###############################################
# construct SVD
svd = TruncatedSVD(n_components=50, random_state=1, algorithm="arpack")

svd.fit((X_train_tfidf))
X_train = svd.fit_transform(X_train_tfidf)
X_test = svd.transform(X_test_tfidf)
print "finished SVD dimension reduction"

###############################################
# save data to file
prepared_data = {'X_train':X_train,
              'y_train':y_train,
              'X_test':X_test,
              'y_test':y_test}

with gzip.open('prepared_data.gz','wb') as f:
  cPickle.dump(prepared_data,f)
f.close()