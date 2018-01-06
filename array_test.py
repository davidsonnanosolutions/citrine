import numpy as np

x = np.arange(10)
y = np.array((0,1,2,3,4,5,6,7,8,9))
g = [0,1,2]
z = [[0],[1],[2]]
print x, y, x.dtype, z
results = [zip(g,z)]
print results

e = np.zeros((10, 1))

print e, e[5]

#accuracy = np.sum(np.int(x == y) for (x, y) in results)
#print accuracy