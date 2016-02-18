from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
             'comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']


twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

length = len(twenty_train.target)

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







