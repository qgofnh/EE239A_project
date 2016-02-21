"""
EE239AS Project 2
Author: Will, Hoooga, Xiaoyu, k

Part b
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
import cPickle,gzip
import re

###############################################
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(txt):
    tokens = re.findall('(?u)\\b\\w\\w+\\b',re.sub('[0-9.]*','',txt.lower()))
    tokens_wo_stop = [item for item in tokens if item not in text.ENGLISH_STOP_WORDS]
    stems = stem_tokens(tokens_wo_stop, stemmer)
    return stems

###############################################
#fetch raw_data
try:
    with gzip.open("raw_data.gz",'rb') as g:
      raw_data=cPickle.load(g)
    g.close()
except:
    print "File not found!"

training_data = raw_data['training']
test_data = raw_data['test']

###############################################
# construct vectorizer and tfidf transformer
vectorizer = CountVectorizer(min_df=1,tokenizer=tokenize)
tfidf = TfidfTransformer(sublinear_tf=True, use_idf=True)

# prepare training data
vectorizer.fit(training_data['data'])
print "finished fitting vectorizer"
X_train_vec = vectorizer.transform(training_data['data'])
print "finished vectorize raw data"

tfidf.fit(X_train_vec)
X_train_tfidf = tfidf.transform(X_train_vec)
print "finished tfidf transform"

y_train = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in training_data['target']]

# prepare testing data
X_test_vec = vectorizer.transform(test_data['data'])
X_test_tfidf = tfidf.transform(X_test_vec)
y_test = [sum([kind == 0, kind == 1, kind == 2, kind == 3]) for kind in test_data['target']]

###############################################
# save data to file
tfidf_data = {'X_train_tfidf':X_train_tfidf,
              'y_train':y_train,
              'X_test_tfidf':X_test_tfidf,
              'y_test':y_test}

with gzip.open('tfidf_data.gz','wb') as f:
  cPickle.dump(tfidf_data,f)
f.close()