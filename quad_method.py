from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def Mid_rule(f,a,b,*args):
    return (b - a) * f((a + b)/2, *args)
    
def Trape_rule(f,a,b,*args):
    return 0.5 * (b - a) * (f(a,*args) + f(b,*args))
    
def Simpson_rule(f,a,b,*args):
    return (b - a) * (f(a,*args) + 4 * f(0.5 * (a + b)) + f(b)) / 6

def Myfun(x):
    return 4/(1 + x**2)

def Composite_quad(f,a,b,type=1,h=0.1,*args):
    '''
    type = 1: midpoint composite quadrature rule
    type = 2: trapezoid composite quadrature rule
    type = 3: Simpson composite quadrature rule
    '''
    integ = 0
    while b>a:
        if b > a + h:
            if type == 1:
                integ = integ + Mid_rule(f,a,a+h,*args)
            elif type == 2:
                integ = integ + Trape_rule(f,a,a+h,*args)
            elif type == 3:
                integ = integ + Simpson_rule(f,a,a+h,*args)
            else:
                print 'Choose the right quadrature rule'
                break
            a = a + h
        else:
            if type == 1:
                integ = integ + Mid_rule(f,a,b,*args)
            elif type == 2:
                integ = integ + Trape_rule(f,a,b,*args)
            elif type == 3:
                integ = integ + Simpson_rule(f,a,b,*args)
            else:
                print 'Choose the right quadrature rule'
                break
            a = b
    return integ

h = np.linspace(0.001,0.1,100)
Mid_error = np.zeros(np.size(h))
Trap_error = np.zeros(np.size(h))
Simpson_error = np.zeros(np.size(h))
i = 0
for step in h:
    Mid_error[i] = np.abs(Composite_quad(Myfun,0,1,type=1,h=step) - np.pi)
    Trap_error[i] = np.abs(Composite_quad(Myfun,0,1,type=2,h=step) - np.pi)
    Simpson_error[i] = np.abs(Composite_quad(Myfun,0,1,type=3,h=step) - np.pi)
    i = i + 1
    
p1 = plt.figure(1)
pp1 = p1.add_subplot(111)
plt.loglog(h, Mid_error, '-', label = 'Mid')
plt.loglog(h, Trap_error, '--', label = 'Trap')
plt.loglog(h, Simpson_error, '-.', label = 'Simpson')
plt.legend(loc='lower right')
plt.xlabel('h')
plt.ylabel('Error')
plt.savefig('PiError.pdf')