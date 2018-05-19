import numpy as np
import scipy.io as sio
import scipy
from scipy.optimize import linprog
from scipy.linalg import norm
from newton_interior_points import mPI_Newton

problem  = 'blend'
solution = -3.0812149846E+01

print('---------------------------------------------',
      'Problema {}:'.format(problem),
      '---------------------------------------------',
      sep='\n')

problem_path = 'test_problems/{}.mat'.format(problem)
file = sio.loadmat(problem_path)
A = file['A']
b = file['b'].flatten()
c = file['c'].flatten()

A = scipy.sparse.csr_matrix.todense(A)
b = b.reshape(b.shape[0],1)
c = c.reshape(c.shape[0],1)

ans = mPI_Newton(A,b,c, warm_initial_solution = False) #See also with warm_initial_solution = True  :) 
print('- - - - - - - - - - - - - - - - - - - - - - -',
      'Nuestra solución: {} '.format(ans['óptimo']),
      'Solución oficial: {}'.format(solution),
      'Número de iteraciones: {}'.format(ans['iter']),
      'Valor de ||diag(x)z||_inf: {}'.format(norm(np.dot(np.diag(ans['x'].T[0]),ans['z']),np.inf)),
      '- - - - - - - - - - - - - - - - - - - - - - -\n',
      sep='\n')
