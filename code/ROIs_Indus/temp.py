import numpy as np
a = np.load("out1.npy")
l = np.array([0,1,2])
idx = list((-a).argsort()[:2])[0][:10]
#print (l[np.argmax(a)])
print (l[np.array(idx)])
