import numpy as np
import pickle as pkl
a = pkl.load(open('ok.pkl', 'rb'))
b = [np.array(w) for w in a]
b[0]
b[0].shape
c= b[0].argmax(-1)
c
c= b[0][:,:,1]
c
c[6]
c[5]
c[5].argmax()
c[5].argsort()
c[5][c[5].argsort()]
b[-1]
b[-1].shape
d = b[-1]
e = b[1]
e
e[5]
d[-1].shape
d[264].sum()
d[265].sum()
d[265].sum()
d[252].sum()
d[253].sum()
[abs(w) for w in d[252]]
sum([abs(w) for w in d[252]])
ccc = [np.sum([abs(w) for w in p]) for p in b[1]]
ccc
b
ccc = [np.sum([abs(w) for w in p]) for p in b[2]]
ccc
np.argsort(ccc)
b[1]
ccc[264]
ccc[265]
ccc[308]
ccc = [np.sum([abs(w) for w in p]) for p in b[2]]
b[2][264]
b[2][308]
ccc = [np.sum([abs(1-w) for w in p]) for p in b[2]]
np.argsort(ccc)
b[1]
bb = np.argsort(ccc).argsort()
bb[0]
newb1 = [[bb[k] for k in p] for p in b[1]]
newb1
ddd = [np.sum([abs(w) for w in p]) for p in b[2]]
dd = np.argsort(ccc).argsort()
newb2 = [[bb[k] for k in p] for p in b[1]]
newb2
newb1
ddd
dd = np.argsort(ddd).argsort()
newb2 = [[dd[k] for k in p] for p in b[1]]
newb2
newb1
ddd = [np.sum([w for w in p]) for p in b[2]]
dd = np.argsort(ddd).argsort()
newb2 = [[dd[k] for k in p] for p in b[1]]
newb2
dd = np.argsort(ddd).argsort()[::-1]
newb2 = [[dd[k] for k in p] for p in b[1]]
newb2
dd = np.argsort(ddd).argsort()
dd = np.argsort(-np.array(ddd)).argsort()
newb2 = [[dd[k] for k in p] for p in b[1]]
newb2
print(newb2)
ccc = [np.sum([1-w for w in p]) for p in b[2]]
cc = np.argsort(-np.array(ccc)).argsort()
newb1 = [[cc[k] for k in p] for p in b[1]]
newb1
