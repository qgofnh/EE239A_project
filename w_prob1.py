from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import pandas as pd

categories_computer = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
categories_recreation = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

com_train = fetch_20newsgroups(subset='train', categories=categories_computer, shuffle=True, random_state=42)
rec_train = fetch_20newsgroups(subset='train', categories=categories_recreation, shuffle=True, random_state=42)
com_test = fetch_20newsgroups(subset='test', categories=categories_computer, shuffle=True, random_state=42)
rec_test = fetch_20newsgroups(subset='test', categories=categories_recreation, shuffle=True, random_state=42)

com_graphics_count = sum(com_train['target'] == 0) + sum(com_test['target'] == 0)
com_misc_count = sum(com_train['target'] == 1) + sum(com_test['target'] == 1)
com_pc_hard_count = sum(com_train['target'] == 2) + sum(com_test['target'] == 2)
com_mac_hard_count = sum(com_train['target'] == 3) + sum(com_test['target'] == 3)
rec_auto_count = sum(com_train['target'] == 0) + sum(com_test['target'] == 0)
rec_motor_count = sum(com_train['target'] == 1) + sum(com_test['target'] == 1)
rec_base_count = sum(com_train['target'] == 2) + sum(com_test['target'] == 2)
rec_hock_count = sum(com_train['target'] == 3) + sum(com_test['target'] == 3)

print "hello"

hist = pd.DataFrame({'count': [com_graphics_count,com_misc_count,com_pc_hard_count,com_mac_hard_count,
                               rec_auto_count,rec_motor_count,rec_base_count,rec_hock_count]})

plt.figure()
hist.plot(kind='hist', alpha=0.5)