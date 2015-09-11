from __future__ import division
import numpy as np
import scipy as sp
import tabulate

def Gaussian_elim_np(A):
    '''
    Perform LU factorization by Gaussian elimination without Partial Pivoting.
    Use the LU = A expression. Return L, U. A is unchanged in the process
    '''
    n = A.shape[0]
    L = np.identity(n)
    U = np.copy(A)
    if n == 1:
        
        print 'Cannot be used for scalar' 
        return 
    for k in xrange(0,n - 1):
        if U[k,k] == 0:
            # For the sake of the printed result, I commented this error message
            # print 'ERROR: pivot is zero'
            continue
        for i in xrange(k + 1,n):
            L[i,k] =  U[i,k] / U[k,k]
            U[i,k] = 0
        for j in xrange(k + 1,n):
            for i in xrange(k + 1, n):
                U[i,j] = U[i,j] - L[i,k] * U[k,j]               
    return L, U

def Gaussian_elim_pp(A):
    '''
    Perform LU factorization by Gaussian elimination with Partial Pivoting(pp).
    A is unchanged in the process.
    Use the LU = PA expression so L is a lower triangular matrix. Return P, L 
    and U. In the process L also need to be permutated. 
    '''
    n = A.shape[0]
    L = np.zeros((n,n)) # Left diagonal 0 because of the permutation later
    P = np.identity(n)
    U = np.copy(A)
    if n == 1:
        print 'Cannot be used for scalar'
        return     
    for k in xrange(0,n - 1):
        # The original index should be +k
        max_idex = np.argmax(np.abs(U[:,k][k:])) + k
        if max_idex != k:
            row_index = np.arange(n)
            row_index[max_idex], row_index[k] = k, max_idex
            U = U[row_index,:]  # Permutate U for partial pivoting
            L = L[row_index,:]  # Permutate L too
            P = P[row_index,:]  # Form the total permutation matrix          
        if U[k,k] == 0:
            print 'ERROR: pivot is zero'
            continue
        for i in xrange(k + 1,n):
            L[i,k] =  U[i,k] / U[k,k]
            U[i,k] = 0
        for j in xrange(k + 1,n):
            for i in xrange(k + 1, n):
                U[i,j] = U[i,j] - L[i,k] * U[k,j]
    np.fill_diagonal(L,1)       # Fill the diagonal of L in the end 
    return P, L, U
     
def Forward_sub(L,b):
    '''
    Forward substitution to solve x of Lx = b. L is lower triangular matrix
    '''
    n = L.shape[0]
    x = np.zeros(n)
    bb = np.copy(b)
    if n != len(bb): 
        print 'ERROR: Dimension does not Match'
        return
    for j in xrange(0,n):
        if L[j,j] == 0:
            # print 'ERROR: Matrix is Singular'
            return
        x[j] = bb[j] / L[j,j]
        for i in xrange(j + 1, n):
            bb[i] = bb[i] - L[i,j] * x[j]
    return x
    
def Back_sub(U,b):
    '''
    Backward substitution to solve x of Ux = b. L is upper triangular matrix
    '''
    n = U.shape[0]
    x = np.zeros(n)
    bb = np.copy(b)
    if n != len(bb): 
        print 'ERROR: Dimension does not Match'
        return
    for j in xrange(n-1,-1,-1):
        if U[j,j] == 0:
            # print 'ERROR: Matrix is Singular'
            return
        x[j] = bb[j] / U[j,j]
        for i in xrange(0,j):
            bb[i] = bb[i] - U[i,j] * x[j]
    return x
    

size = np.array([5,10,100])

headers = ['Algorithm','Non-pivoting','Partial Pivoting ','np.linalg.solve']
print 'Result of the Type I Matrix'
for n in size:
    error_list = ['Error of solution']
    residual_list = ['Residual']
    A = np.random.rand(n,n)   
    x = np.ones(n)
    b = np.dot(A,x)
    cond_num = np.linalg.cond(A,np.inf)
    
    # Gaussian elimination with no pivoting
    L1, U1 = Gaussian_elim_np(A)
    x1 = Back_sub(U1,Forward_sub(L1,b))
    if x1 == None:
        error_list.append('FAIL!')
        residual_list.append('FAIL!')
    else:
        error_list.append(np.max(np.abs(x1 - x)))
        residual_list.append(np.max(np.abs(np.dot(A,x1) - b)))
    # Gaussian elimination with partial pivoting
    P2, L2, U2 = Gaussian_elim_pp(A)
    x2 = Back_sub(U2,Forward_sub(L2,np.dot(P2,b)))
    if x2 == None:
        error_list.append('FAIL!')
        residual_list.append('FAIL!')
    else:
        error_list.append(np.max(np.abs(x2 - x)))
        residual_list.append(np.max(np.abs(np.dot(A,x2) - b)))
    # Gaussian elimination with build-in solver
    x3 = np.linalg.solve(A,b)
    P4, L4, U4 = sp.linalg.lu(A)
    if x3 == None:
        error_list.append('FAIL!')
        residual_list.append('FAIL!')
    else:
        error_list.append(np.max(np.abs(x3 - x)))
        residual_list.append(np.max(np.abs(np.dot(A,x3) - b)))
    table = [error_list,residual_list]
    print 'Matrix of Size ' + str(n) +'; Condition Number: ' + str(cond_num)
    print tabulate.tabulate(table, headers, tablefmt="rst")
    print
print

print 'Result of the Type II Matrix'
for n in size:
    error_list = ['Error of solution']
    residual_list = ['Residual']
    A = np.zeros((n,n))  
    for i in xrange(n):
        for j in xrange(n):
            A[i,j] = 2 * max(i,j) + 1
  
    x = np.ones(n)
    b = np.dot(A,x)
    cond_num = np.linalg.cond(A,np.inf)
    
    # Gaussian elimination with no pivoting
    L1, U1 = Gaussian_elim_np(A)
    x1 = Back_sub(U1,Forward_sub(L1,b))
    if x1 == None:
        error_list.append('FAIL!')
        residual_list.append('FAIL!')
    else:
        error_list.append(np.max(np.abs(x1 - x)))
        residual_list.append(np.max(np.abs(np.dot(A,x1) - b)))
    # Gaussian elimination with partial pivoting
    P2, L2, U2 = Gaussian_elim_pp(A)
    x2 = Back_sub(U2,Forward_sub(L2,np.dot(P2,b)))
    if x2 == None:
        error_list.append('FAIL!')
        residual_list.append('FAIL!')
    else:
        error_list.append(np.max(np.abs(x2 - x)))
        residual_list.append(np.max(np.abs(np.dot(A,x2) - b)))
    # Gaussian elimination with build-in solver
    x3 = np.linalg.solve(A,b)
    P4, L4, U4 = sp.linalg.lu(A)
    if x3 == None:
        error_list.append('FAIL!')
        residual_list.append('FAIL!')
    else:
        error_list.append(np.max(np.abs(x3 - x)))
        residual_list.append(np.max(np.abs(np.dot(A,x3) - b)))
    table = [error_list,residual_list]
    print 'Matrix of Size ' + str(n) +'; Condition Number: ' + str(cond_num)
    print tabulate.tabulate(table, headers, tablefmt="rst")
    print
print

print 'Result of the Type III Matrix'
for n in size:
    error_list = ['Error of solution']
    residual_list = ['Residual']
    A = np.zeros((n,n))
    A.fill(0.001)
    for i in xrange(n):
        for j in xrange(n):
            if (i + 2) % n == j:
                A[i,j] = 5
  
    x = np.ones(n)
    b = np.dot(A,x)
    cond_num = np.linalg.cond(A,np.inf)
    
    # Gaussian elimination with no pivoting
    L1, U1 = Gaussian_elim_np(A)
    x1 = Back_sub(U1,Forward_sub(L1,b))
    if x1 == None:
        error_list.append('FAIL!')
        residual_list.append('FAIL!')
    else:
        error_list.append(np.max(np.abs(x1 - x)))
        residual_list.append(np.max(np.abs(np.dot(A,x1) - b)))
    # Gaussian elimination with partial pivoting
    P2, L2, U2 = Gaussian_elim_pp(A)
    x2 = Back_sub(U2,Forward_sub(L2,np.dot(P2,b)))
    if x2 == None:
        error_list.append('FAIL!')
        residual_list.append('FAIL!')
    else:
        error_list.append(np.max(np.abs(x2 - x)))
        residual_list.append(np.max(np.abs(np.dot(A,x2) - b)))
    # Gaussian elimination with build-in solver
    x3 = np.linalg.solve(A,b)
    P4, L4, U4 = sp.linalg.lu(A)
    if x3 == None:
        error_list.append('FAIL!')
        residual_list.append('FAIL!')
    else:
        error_list.append(np.max(np.abs(x3 - x)))
        residual_list.append(np.max(np.abs(np.dot(A,x3) - b)))
    table = [error_list,residual_list]
    print 'Matrix of Size ' + str(n) +'; Condition Number: ' + str(cond_num)
    print tabulate.tabulate(table, headers, tablefmt="rst")
    print
print