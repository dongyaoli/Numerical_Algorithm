from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npla

def Euler_forward(f,y0,t0,t1,h=0.01):
    n = int((t1 - t0)/h)
    y = np.zeros(n+1)
    t = np.zeros(n+1)
    t[0] = t0
    y[0] = y0
    for i in xrange(1,n+1):
        y0 = y0 + h * f(y0, t0)
        t0 = t0 + h
        y[i] = y0
        t[i] = t0
    return t, y

def Jacobian(f, x0, *args):
    n = np.size(x0)
    h = 10**-8
    Jacobi = np.zeros((n,n))
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = h
        Jacobi[:,i] = (f(x0 + delta, *args) - f(x0 - delta, *args))/(2 * h)
    return Jacobi

def Newton(F, x0, *args):
    tol=10**-12
    x = x0
    count = 0
    while True:
        count += 1
        J = Jacobian(F, x, *args)
        s = npla.solve(J,-F(x, *args))
        if np.max(np.abs(s)) < tol:
            break
        x = x + s
    return x

def Euler_backward(f,y0,t0,t1,h=0.01):
    
    def implicit_fun(y1,y0,t1,f):
        return h * f(y1, t1) + y0 - y1
        
    n = int((t1 - t0)/h)
    y = np.zeros(n+1)
    t = np.zeros(n+1)
    t[0] = t0
    y[0] = y0
    for i in xrange(1,n+1):
        y_guess = y0 + h * f(y0, t0)
        t0 = t0 + h
        y1 = Newton(implicit_fun, y_guess, y0, t0, diff)
        y[i] = y1
        t[i] = t0
        y0 = y1
    return t, y

def Taylor_Method(f,y0,t0,t1,h=0.01):
    def second_deri(f,y,t):
        delta = 10**-8
        ft = (f(y,t + delta) - f(y,t - delta))/(2 * delta)
        fy = (f(y + delta,t) - f(y - delta,t))/(2 * delta)
        return ft + fy * f(y,t)
       
    n = int((t1 - t0)/h)
    y = np.zeros(n+1)
    t = np.zeros(n+1)
    t[0] = t0
    y[0] = y0
    for i in xrange(1,n+1):
        y0 = y0 + h * f(y0, t0) + h * h * second_deri(f,y0,t0)/2
        t0 = t0 + h
        y[i] = y0
        t[i] = t0
    return t, y

def diff(y,t):
    return -y + 1/(1 + t**2) + np.arctan(t)

step = np.array([0.5,0.1,0.01,0.001])

t0 = np.linspace(0,5,100)

p1 = plt.figure(1)
pp1 = p1.add_subplot(111)
for h0 in step:
    t1, y1 = Euler_forward(diff,0,0,5,h=h0)
    plt.plot(t1, y1, label = 'h=' + str(h0))
plt.legend(loc='lower right')
pp1.set_title('Euler Forward')
plt.xlabel('t')
plt.ylabel('y')
plt.savefig('Euler_forward.pdf')

p2 = plt.figure(2)
pp2 = p2.add_subplot(111)
for h0 in step:
    t1, y1 = Euler_backward(diff,0,0,5,h=h0)
    plt.plot(t1, y1, label = 'h=' + str(h0))
plt.legend(loc='lower right')
pp2.set_title('Euler Backward')
plt.xlabel('t')
plt.ylabel('y')
plt.savefig('Euler_backward.pdf')

p3 = plt.figure(3)
pp3 = p3.add_subplot(111)
for h0 in step:
    t1, y1 = Taylor_Method(diff,0,0,5,h=h0)
    plt.plot(t1, y1, label = 'h=' + str(h0))
plt.legend(loc='lower right')
pp3.set_title('Taylor Method')
plt.xlabel('t')
plt.ylabel('y')
plt.savefig('Taylor.pdf')

p4 = plt.figure(4)
pp4 = p4.add_subplot(111)
error1 = []
error2 = []
error3 = []
for h0 in step:
    t1, y1 = Euler_forward(diff,0,0,5,h=h0)
    t2, y2 = Euler_backward(diff,0,0,5,h=h0)
    t3, y3 = Taylor_Method(diff,0,0,5,h=h0)
    error1.append(np.abs(y1[-1] - np.arctan(t1[-1])))
    error2.append(np.abs(y2[-1] - np.arctan(t2[-1])))
    error3.append(np.abs(y3[-1] - np.arctan(t3[-1])))
plt.loglog(step,error1,'-', label = 'Euler Forward')    
plt.loglog(step,error2,'--', label = 'Euler Backward')
plt.loglog(step,error3,'-.', label = 'Taylor Series')
plt.legend(loc='lower right')
plt.xlabel('h')
plt.ylabel('Error')
pp4.set_title('Error')
plt.savefig('error_v_h.pdf')