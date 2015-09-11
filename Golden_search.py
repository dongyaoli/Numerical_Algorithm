from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def Golden_search(f, a, b, tol=10**-16):
    tau = (5**0.5 - 1)/2
    x1 = a + (1 - tau) * (b - a)
    f1 = f(x1)
    x2 = a + tau * (b - a)
    f2 = f(x2)
    while ((b - a) > tol):
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + tau * (b -a)
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - tau) * (b - a)
            f1 = f(x1)
    return x1

def myfun(x):
    return 120 * x + (50 + 30 * np.exp(-100 * x)) * (1 - x)

brac = np.arange(-0.02,0.5,0.0001)
ax = plt.figure(1)
f1 = ax.add_subplot(111)
curve, = plt.plot(brac,myfun(brac))
minimum = Golden_search(myfun, -0.02, 0.5)
mini, = plt.plot(minimum,myfun(minimum),'ro', label='Minimum')
plt.legend(loc='upper right')
plt.savefig('Golden_search.png')
print 'The minimum is located at x = ' + str(minimum)