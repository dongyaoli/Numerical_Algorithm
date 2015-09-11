from __future__ import division
import numpy as np
import numpy.linalg as npla

def Jacobian(f, x0):
    n = np.size(x0)
    h = 10**-8
    Jacobi = np.zeros((n,n))
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = h
        Jacobi[:,i] = (f(x0 + delta) - f(x0 - delta))/(2 * h)
    return Jacobi

def Newton(F, x0, tol=10**-16):
    x = x0
    count = 0
    while True:
        count += 1
        J = Jacobian(F, x)
        s = npla.solve(J,-F(x))
        if np.max(np.abs(s)) < tol:
            break
        x = x + s
    return x, count

def Broyden(F, x0, tol=10**-16):
    x = x0
    # Use true Jacobina as initial guess
    B = Jacobian(F, x)
    count = 0
    while True:
        s = npla.solve(B, -F(x))
        ss = s[np.newaxis,:]
        count += 1
        if np.max(np.abs(s)) < tol:
            break       
        y = F(x + s) - F(x)
        x = x + s
        B = B + (np.dot((y - np.dot(B,s))[np.newaxis,:].T, ss)/np.dot(ss,ss.T))
    return x, count

def nl_system(x):
    n = np.size(x)
    if n != 2:
        print 'Wrong input for the function!'
        return
    f = np.zeros(2)
    f[0] = (x[0] + 3) * (x[1]**3 - 7) + 18
    f[1] = np.sin(x[1] * np.exp(x[0]) - 1)
    return f

x0 = np.array([-0.5, 1.4])
x1, count1 = Newton(nl_system, x0)
x2, count2 = Broyden(nl_system, x0)
print 'Newton\'s method gives result: ' + str(x1) + ' in ' + str(count1) + \
' iteration'
print 'Broyden\'s method gives result: ' + str(x2) + ' in ' + str(count2) + \
' iteration'