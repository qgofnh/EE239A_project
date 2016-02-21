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
from nltk.stem.porter import PorterStemmer
import string
import re
from nltk import SnowballStemmer
import numpy as np
import cPickle,gzip

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(txt):
    # txt = "".join([ch for ch in txt if ch not in string.punctuation])
    # tokens = word_tokenize(txt)
    # tokens = [i for i in tokens if i not in string.punctuation]

    tokens = re.findall('(?u)\\b\\w\\w+\\b',re.sub('[0-9.]*','',txt.lower()))
    tokens_wo_stop = [item for item in tokens if item not in text.ENGLISH_STOP_WORDS]
    stems = stem_tokens(tokens_wo_stop, stemmer)
    return stems


################### fetch data
categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
training_data = fetch_20newsgroups()

t = training_data['data'][0]
print t
################### clean data
dt = training_data['data']
typ = training_data['target']
dt_typ = dict(zip(dt,typ))
X_data = ['']*20                                        # put all documents in the same class together
for data,type in dt_typ.items():
    X_data[type] = X_data[type] + data

X_tokens = [tokenize(item) for item in X_data]

count = [dict(),dict(),dict(),dict()]
i = 0
for k in [3,4,6,15]:
    for item in X_tokens[k]:
        count[i][item] = count[i].get(item,0) + 1
    i += 1

l = []
for term,c in count[2].items():
    l.append((c,term))

l.sort(reverse=True)

ls = []
for c,t in l:
    ls.append(((.5+.5*c/l[0][0])*np.log(4/(np.asarray([t in item for item in X_tokens]).sum())),t))
    # print t,np.asarray([t in item for item in X_tokens]).sum()


ls.sort(reverse=True)
# np.asanyarray(count[1].values()).sum()
print "done"

with gzip.open('class-2_top10.gz','wb') as f:
  cPickle.dump(ls,f)
f.close()