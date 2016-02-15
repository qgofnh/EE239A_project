# Proj2 a)
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt
import numpy as np
categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
a = [len(fetch_20newsgroups(subset='train', categories=[i], shuffle=True, random_state=42).data) for i in categories]



#################################### a) plot histogram
plt.bar(np.arange(len(a)), a, 0.9, align='center', alpha=0.5)
plt.xticks(np.arange(len(a)),categories,rotation = 20,ha = 'right')
plt.xlabel('Categories')
plt.ylabel('Count')
# plt.show()
#################################### plt.show()


#################################### b) remove stop word
stop_words = text.ENGLISH_STOP_WORDS
print len(stop_words)

# print a
plt.bar(np.arange(len(a)), a, 0.5, align='center', alpha=0.5)
plt.xticks(np.arange(len(a)),categories,rotation=20,ha = 'right')
plt.show()
plt.xlabel('Categories')
plt.ylabel('Count')


