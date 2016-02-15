# Proj2 a)
from sklearn.datasets import fetch_20newsgroups
from StdSuites.Type_Names_Suite import rotation
import numpy as np
categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

docs = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
a = [np.sum(docs.target == i) for i in range(8)]


a = [len(fetch_20newsgroups(subset='train', categories=[i], shuffle=True, random_state=42).data) for i in categories]

print a

hist = {categories[i]:[a[i]] for i in range(8)}

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# plt.figure()
# plt.bar(np.arange(len(a)), a, 0.9, align='center', alpha=0.5)
# plt.xticks(np.arange(len(a)),categories,rotation = 20,ha = 'right')
# plt.show()

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
X = [vectorizer.fit_transform(i) for i in docs.data[i]]











