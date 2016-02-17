# Proj2 a)
from sklearn.datasets import fetch_20newsgroups
categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

a = [len(fetch_20newsgroups(subset='train', categories=[i], shuffle=True, random_state=42).data) for i in categories]
print a









