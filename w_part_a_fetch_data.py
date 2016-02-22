"""
EE239AS Project 2
Author: Will, Hoooga, Xiaoyu, k

Part a
"""
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import pandas as pd
import cPickle,gzip

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']


training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

counts = [sum(training_data['target'] == i) for i in range(8)]

hist = pd.DataFrame({'count': counts})
hist.plot(kind='bar',alpha=0.5)

raw_data = {'training':training_data,'test':test_data}

with gzip.open('raw_data.gz','wb') as f:
  cPickle.dump(raw_data,f)
f.close()