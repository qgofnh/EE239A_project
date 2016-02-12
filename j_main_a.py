# Proj2 a)
from sklearn.datasets import fetch_20newsgroups
categories_computer = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
categories_recreation = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

com_train = fetch_20newsgroups(subset='train', categories=categories_computer, shuffle=True, random_state=42)
rec_train = fetch_20newsgroups(subset='train', categories=categories_recreation, shuffle=True, random_state=42)
com_test = fetch_20newsgroups(subset='test', categories=categories_computer, shuffle=True, random_state=42)
rec_test = fetch_20newsgroups(subset='test', categories=categories_recreation, shuffle=True, random_state=42)

data1 = fetch_20newsgroups(subset='train', categories=['comp.graphics'], shuffle=True, random_state=42)

data2 = fetch_20newsgroups(subset='train', categories=['comp.os.ms-windows.misc'], shuffle=True, random_state=42)

data3 = fetch_20newsgroups(subset='train', categories=['comp.sys.ibm.pc.hardware'], shuffle=True, random_state=42)

data4 = fetch_20newsgroups(subset='train', categories=['comp.sys.mac.hardware'], shuffle=True, random_state=42)

