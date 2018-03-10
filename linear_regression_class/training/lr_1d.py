import numpy as np
import matplotlib.pyplot as plt

# load data
X = []
Y = []
for line in open('../data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# let's turn x and y into numpy arrays
X = np.array(X)
Y = np.array(Y)

# plot to see what it looks like
plt.scatter(X, Y)
plt.show()

# computation
N = len(X)
C = np.dot(X, X)
D = X.sum()
E = np.dot(X, Y)
F = Y.sum()

determinant = N * C - D * D

a = (N * E - F * D) / determinant
b = (E * D - F * C) / (-determinant)

print(N)
print("")

print(C)
print(D)
print(E)
print(F)
print("")

print (a)
print (b)

# Calculate predicted Y
Yhat = a * X + b

# plot it all
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# compute R2
SSres = Y - Yhat
SSres = SSres.dot(SSres)
SStot = Y - Y.mean()
SStot = SStot.dot(SStot)
R2 = 1 - (SSres / SStot)

print(R2)
