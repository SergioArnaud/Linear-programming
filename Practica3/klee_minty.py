import numpy as np
from numpy import concatenate, array , transpose, dot, diag, inf, ones, eye, zeros
from newton_interior_points import mPI_Newton

def genera_klee_minty(m):
    c =- ones(m)
    A = zeros((m,m),dtype=float)
    b = array([(2**(i+1))-1 for i in range(m)],dtype=float)
    
    for i in range(m):
        A[i,i]=1
        for j in range(i):
            A[i,j]=2
            
    A=np.hstack((A,np.eye(m)))
    c=np.concatenate((c,np.zeros(m)))
            
    return(A,b,c)


m=[10,12,14,16]

for i in m:
    A,b,c=genera_klee_minty(i)

    b = b.reshape(b.shape[0],1)
    c = c.reshape(c.shape[0],1)
    
    ans = mPI_Newton(A, b, c, warm_initial_solution=True, show_iteration_and_norm = False)
    print('-----------------------------------------------------------------')
    
    print('Klee-Minty con m =',i, "(iniciando con una 'warm solution')")
    print('Nuestra solución:',ans['óptimo'])
    print('Solución teorica:',-((2**i) -1))
    print('Número de iteraciones:', ans['iter'])
    
    ans = mPI_Newton(A, b, c, show_iteration_and_norm = False)
    print('\nKlee-Minty con m =',i, "(sin iniciar con una 'warm solution')")
    print('Nuestra solución:',ans['óptimo'])
    print('Solución teorica:',-((2**i) -1))
    print('Número de iteraciones:', ans['iter'])
    
    print('\nSolución al primal (x):\n', ans['x'][:i].T)
    print('Solución al dual (z):\n', ans['z'][:i].T)
    print('Solución al dual (lambda):\n', ans['lambda_'][:i].T)
    
    print('-----------------------------------------------------------------\n')