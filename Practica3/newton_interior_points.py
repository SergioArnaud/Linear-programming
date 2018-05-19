import numpy as np
from numpy import concatenate, array , transpose, dot, diag, inf, ones, eye, zeros
from scipy.linalg import solve as solve_linear_system
from scipy.linalg import lstsq, norm
from scipy.optimize import linprog

def recorte(v,dv):
    '''
    Function that retuns the maximum value 0 < alpha <= 1 such that v + alpha*dv >= 0

    '''

    alpha = 1
    for v_i, dv_i in zip(v,dv):
        if dv_i[0] < 0:
            alpha = min(alpha,-v_i[0]/dv_i[0])

    return alpha
    
    
def mPI_Newton(A, b, c, tol=1e-9, sigma = .1, warm_initial_solution=False, show_iteration_and_norm = True):

    '''

        Given the lineal program:
        
        min { c^T x }
         x
            Ax <= b,
            x >= 0,
            b >= 0

        The algorithm solves the primal and dual problems using a Newton interior point method.

        Input: 

        A:                      (matrix)
        b:                      (vector)
        c:                      (vector)
        tol:                    (float) the error tolerance

        warm_initial_solution:   (bool) If set True the algorithm uses the square minimum aproximations of
                                       x, lambda_ and z (with some modifications) to begin the convergence, 
                                       if not it uses the following default values x = ones((n,1)), lambda_ = zeros((m,1))
                                        z = ones((n,1)) 

        show_iteration_and_norm: (bool) If set True the algorithm prints the number of iteration and the norm of F (||F||)
                                        at every step


    '''
    
    num_iter = 0
    m,n = A.shape[0], A.shape[1]
    
    def F():
        
        row_1 = np.dot(A,x)-b
        row_2 = np.dot(A.T,lambda_) + z - c
        row_3 = np.multiply(x,z)
        
        return np.concatenate([row_1, row_2, row_3 ])
    
    def build_jacobian_newton():
    
        X = np.diag(x.T[0])
        Z = np.diag(z.T[0])
        
        col_1 = concatenate([A,zeros((n,n)),Z])
        col_2 = concatenate([zeros((m,m)), A.T, zeros((n,m)) ])
        col_3 = concatenate([zeros((m,n)), eye(n), X])

        return np.hstack((col_1, col_2, col_3))

    def get_initial_parameters():
        
        if warm_initial_solution:
            x = lstsq(A,b)[0]
            lambda_ = lstsq(A.T,c)[0]
            z = c - np.dot(A.T,lambda_)

            x = x + np.max([0, -3/2*np.min(x)])*np.ones((n,1))
            z = z + np.max([0, -3/2*np.min(z)])*np.ones((n,1))

            if np.min(x) == 0:
                mu = np.dot(x.T,z)/2
                x = x + mu/np.dot(np.ones((m,1)).T,z)*np.ones((n,1))
                z = z + mu/np.dot(np.ones((n,1)).T,x)*np.ones((m,1))
        
        else:
            x = ones((n,1))
            lambda_ = zeros((m,1))
            z = ones((n,1))
        
        return array(x), array(lambda_), array(z)
    
    
    x, lambda_, z = get_initial_parameters()
    F_w = F()
    error = norm(F_w, inf)
    
    while error > tol and num_iter < 200:
        
        mu = dot(x.T,z)/n
        Jac = build_jacobian_newton()
        
        aux = concatenate( [zeros((m,1)), zeros((n,1)), sigma*mu*ones((n,1))] )
        delta = lstsq(Jac, -F_w + aux)[0]
        
        delta_x = delta[:n]
        delta_l = delta[n:n+m]
        delta_z = delta[n+m:]
        
        alpha_x = recorte(x,delta_x)
        alpha_z = recorte(z,delta_z)
    
        x = x + (999/1000)*alpha_x*delta_x
        lambda_ = lambda_ + (999/1000)*alpha_z*delta_l
        z = z + (999/1000)*alpha_z*delta_z
        
        if alpha_z*alpha_x > 0.8:
            sigma = np.max([sigma/10, 10e-4])
            
        F_w = F()
        error = norm(F_w, inf)
        num_iter += 1
        
        if show_iteration_and_norm:
            print('Iteración número:', num_iter)
            print('Norma de F(wk):  ', error,'\n')
        

    return  {
                'óptimo'  : dot(transpose(c),x)[0][0],
                'x'       : x,
                'z'       : z,
                'lambda_' : lambda_,
                'iter'    : num_iter
            }
