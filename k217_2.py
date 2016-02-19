from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer


# tokenizer

stemmer = SnowballStemmer('english')
def stem_tokens(tokens,stemmer):
    stemmed=[]
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens,stemmer)
    return stems

##########################################

categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
             'comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']


twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

##############################################
# problem 1
length = len(twenty_train.target)
print ('length',length)

num_list = [0]*8

for i in range(0,length-1):
    a = twenty_train.target[i]
    num_list[a]+=1
print num_list

# use pandas dataframe
d = {'number':pd.Series(num_list,index=categories)}
df=pd.DataFrame(d)

plt.figure()
df.plot(kind='bar',alpha=0.5)
#plt.show()

#################################################
# problem 2
vectorizer = CountVectorizer(min_df=1, stop_words=text.ENGLISH_STOP_WORDS, tokenizer=tokenize)

X_train_counts = vectorizer.fit_transform(twenty_train.data)

print('original terms have: 79218')
print('take out stop words: 78911')

print('the new size after vectorizer: 106753')
print(X_train_counts.shape)

###################################################
# TF-iDF
tfidf_transformaer= TfidfTransformer()
X_train_tfidf = tfidf_transformaer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)
print('feature of tf-idf: 106753')




