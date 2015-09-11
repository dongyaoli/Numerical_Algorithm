from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npla

def Jacobian(f, x0, *args):
    """
    Parameters
    ----------
    f : the function which to calculate Jacobian of
    x0 : argument to perform derivative
    args: other arguments of the function
    """  
    n = np.size(x0)
    m = np.size(f(x0, *args))
    h = 10**-8
    Jacobi = np.zeros((m,n))
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = h
        Jacobi[:,i] = (f(x0 + delta, *args) - f(x0 - delta, *args))/(2 * h)
    return Jacobi



def residu(x, y, f, *args):
    return y - f(x, *args)

def Gauss_Newton(f, t, y, x, tol=10**-10):
    """
    Parameters
    ----------
    f : Target function to perform fitting
    t : first coordinate of data points
    y : second coordinate of data points
    x : parameters to fit
    """
    while True:
        resi = residu(x, y, f, t)
        A = Jacobian(residu, x, y, f, t)
        q, r = npla.qr(A)
        s0 = npla.solve(r, np.dot(q.T, -resi)) 
        if np.max(np.abs(s0)) < tol:
            break
        x = x + s0
    return x
    
def myfun(x, t):
    return x[0] * np.exp(x[1] * t)    
    
t_data = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
y_data = np.array([6.8, 3, 1.5, 0.75, 0.48, 0.25, 0.2, 0.15])    

x0 = np.array([1,1])
x1 = Gauss_Newton(myfun, t_data, y_data, x0)
t_curve = np.arange(0, 5, 0.1)
y_curve1 = myfun(x1, t_curve)
pt1 = plt.figure(1).add_subplot(111)
curve, = plt.plot(t_data, y_data, 'ro', label = 'data')
fit = plt.plot(t_curve, y_curve1, label = 'fitting')
plt.legend(loc='upper right')
pt1.set_title('Nonlinear fit using Gauss-Newton method')
print 'coefficient obtained by Gauss-Newton method is: ' + str(x1)
plt.savefig('Nonlinear_Fit.pdf')

y_log_data = np.log(y_data)
A2 = np.array([np.ones(np.size(t_data)), t_data]).T
q2, r2 = npla.qr(A2)
x_log = npla.solve(r2, np.dot(q2.T, y_log_data))
x2 = np.array([np.exp(x_log[0]), x_log[1]])
pt2 = plt.figure(2).add_subplot(111)
curve, = plt.plot(t_data, y_data, 'ro', label = 'data')
y_curve2 = myfun(x2, t_curve)
fit = plt.plot(t_curve, y_curve2, label = 'fitting')
plt.legend(loc='upper right')
pt2.set_title('Linear fit after taking logarithm')
print 'coefficient obtained by linear fitting is: ' + str(x2)
plt.savefig('Linear_Fit.pdf')