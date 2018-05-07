import numpy as np
from numpy import hstack, array, eye, zeros, arange, transpose, dot, inf
import scipy 
from scipy.linalg import inv
from scipy.linalg import solve as solve_linear_system
from collections import namedtuple


def mSimplexDual(A,b,c):
    '''
    Dual-Simplex algorithm.
    
        min { c^T x }
         x
            Ax <= b,
            x >= 0,
            c >= 0
            
    Input: 
        A: matrix
        b: vector
        c: vector
        
    Output:
        x_0: A basic feasible solution to the problem
        
        z_0: The solution of the optimization problem
        
        ban: Several cases:
                            -1 :    Feasible set is empty
                            0  :    Optimal value found
                            1  :    Non-bounded problem
             
        iter: Number of iterations (changes of basic variables)
        
        lamo: Solution to the dual problem
    '''
    
    m = A.shape[0]
    n = A.shape[1]
    
    A = hstack((A,-eye(m)))
    c=np.concatenate((c,zeros(m)))
    n = n+m
    
    B = [i for i in range(n-m,n)]
    
    iteraciones=0
    while iteraciones < 500000:
        
        N = [i for i in range(n) if not i in B]
        x_B = solve_linear_system(A[:,B],b)
        
        if min(x_B) >= 0:
            
            x = array([0. for i in range(n-m)])
            
            for i in range(m):
                if B[i] < n-m :
                    x[B[i]] = x_B[i]
                    
            sol_dual = solve_linear_system( transpose(A[:,B]), c[B] )
            
            return {
                    'x_0' : x,
                    'z_0' : np.dot(c[B],x_B),
                    'ban' : 0,
                    'iter': iteraciones,
                    'lamo': sol_dual
                    }
        
        salida = -1
        for i in range(m):
            if x_B[i]<0:
                salida=i
                break
        
        H = solve_linear_system(A[:,B], A)
        
        r_N = -dot(c[B], H[:,N]) + c[N]
        
        entrada = -1
        for i in range(n-m):
            if H[salida][N[i]]<0:
                if entrada == -1:
                    entrada = i
                elif r_N[i]/H[salida][N[i]] > r_N[entrada]/H[salida][N[entrada]]:
                    entrada=i
                    
        if entrada == -1:
            return {
                    'x_0' : None,
                    'z_0' : None,
                    'ban' : -1,
                    'iter': iteraciones,
                    'lamo': None
                    }
        
        B[salida] = N[entrada]
                
        iteraciones = iteraciones + 1
        
    print('Número máximo de iteraciones alcanzado')
        



def mSimplexMax(A, b, c):
    """
    Phase II, Simplex algorithm.
    
        max { c^T x }
         x
            Ax <= b,
            x >= 0,
            b >= 0

    Input: 
        A: matrix
        b: vector
        c: vector

    Output: 

        A JSON object containing the following keys:

            x0:         A basic feasible solution or a ascent direction
                        in case the problem is not bounded
            
            z0:         The solution of the optimization problem,
                        if the problem is not bounded returns inf 
            
            ban:        Several cases:
                            -1 :    Feasible set is empty
                            0  :    Optimal value found
                            1  :    Non-bounded problem
                
            iter:       Number of iterations (changes of basic variables)
            
            sensinfo:   (When an optimal value was found):
                 
                 sensinfo.lambda : Dual solution
                 sensinfo.gammas : 2 x m matrix 
                 sensinfo.betas  : 2 x n matrix 
    """
    
    c = -c
    ans = mSimplexMin(A,b,c)
    
    ans['z0'] = - ans['z0']
    sensinfo = ans['sensinfo']
    
    for i in sensinfo.gammas:
        temp = -i[0]
        i[0] = -i[1]
        i[1] = temp
        
    for i in range(len(sensinfo.lambda_)):
        sensinfo.lambda_[i] = - sensinfo.lambda_[i]
    
    ans['sensinfo'] = sensinfo
    
    return ans
    

def mSimplexMin(A,b,c, max_iter = 100000):

    """
    Phase II, Simplex algorithm.
    
        min { c^T x }
         x
            Ax <= b,
            x >= 0,
            b >= 0

    Input: 
        A: matrix
        b: vector
        c: vector

    Output: 

        A JSON object containing the following keys:

            x0:     A basic feasible solution or a descent direction
                    in case the problem is not bounded 
            
            z0:     The solution of the optimization problem,
                    if the problem is not bounded returns -inf 
            
            ban:    Several cases:
            
                    -1  Feasible set is empty
                    0   Optimal value found
                    1   Non-bounded problem
                
            iter:   Number of iterations
            
            sensinfo:   (When an optimal value was found):
                 
                         sensinfo.lambda : Dual solution
                         sensinfo.gammas : 2 x m matrix 
                         sensinfo.betas  : 2 x n matrix

    """

    len_b = len(b)
    len_c = len(c)

    A = hstack( (A, eye(len_b)) )
    c = hstack( (c, zeros(len_b)) )
     
    N = np.arange(0, A.shape[1]-len_b)
    B = np.arange(A.shape[1]-len_b, A.shape[1])
    
    num_iter = 0
    while num_iter < max_iter:
        ans = simplex_step(A, b, c, B, N, len_b, len_c)
        
        if ans['Status'] == "Solution found":

            ans =   {
                    'x0': ans['BFS'][0:len_c],
                    'z0': ans['Optimal Value'], 
                    'ban': 0, 
                    'iter': num_iter ,
                    'sensinfo': ans['sensinfo']
                    }
            return ans

        elif ans['Status'] == 'Not bounded problem':

            ans =   {
                    'x0': ans['Descent direction'],
                    'z0': -inf, 
                    'ban': 1, 
                    'iter': num_iter 
                    }
            
            return ans
 
        num_iter += 1

        B = ans['B']
        N = ans['N']
    
    return 'Número máximo de iteraciones alcanzado'

def get_gammas(N, r_N, H, m, n ):
    
    """
    Function that returns a matrix with the intervals for the gammas

    gammas : matrix with the form:

             [ [g00, g01, g02, ..., g0m],
               [g10, g11, g12, ..., g1m] ]

        where:

            [g0j, g1j] represents the interval for the jth gamma
                                            
    """
    
    gammas = zeros([m,2])
    for j in range(m):
        
        if j in N:
            a,b = -inf, r_N[j]
        
        else:
            A,B = [-inf],[inf]
            
            A.extend([-r_N[k] / H[j,k] for k in range(m) if H[j,k] < 0])
            B.extend([-r_N[k] / H[j,k] for k in range(m) if H[j,k] > 0])
            
            a,b = max(A), min(B)
            
        gammas[j,0] = a
        gammas[j,1] = b
            
    return gammas




    
def get_betas(x_B, A_B_inv, n, m):
    
    """
    Function that returns a matrix with the intervals for the betas

    gammas : matrix with the form:

             [ [b00, b01, b02, ..., b0n],
               [b10, b11, b12, ..., b1n] ]

        where:

            [b0j, b1j] represents the interval for the jth beta
                                            
    """
    
    betas = zeros([n,2])
    for j in range(n):
        A,B = [-inf],[inf]
        
        B.extend([-x_B[k] / A_B_inv[k,j] for k in range(n) if A_B_inv[k,j] < 0])
        A.extend([-x_B[k] / A_B_inv[k,j] for k in range(n) if A_B_inv[k,j] > 0])
        
        a,b = max(A), min(B)
        
        betas[j,0] = a
        betas[j,1] = b
    
    return betas


def simplex_step(A, b, c, B, N, len_b, len_c):

    """
    Function that performs one step of the simplex algorithm updating the
    basic and the non-basic variables

    Input:

        A: matrix
        b: vector
        c: vector
        B: vector
           indexes of the basic variables
        N: vector
           indexes of the non-basic variables

    Output:
    
        A JSON object that can contain the following keys:

            Status          : Several cases:
                                    'Solution found'
                                    'Not bounded problem'
                                    'Method continues'
                                    
            Optimal Value   : The solution of the optimization problem,
                              if the problem is not bounded returns -inf
            
            BFS             : A basic feasible solution or a descent direction
                              in case the problem is not bounded
            
            N               : Array with the non-basic variables
            
            B               : Array with the basic variables

    """

    B = sorted(B)
    N = sorted(N)
    
    c_B = array([c[i] for i in B])
    c_N = array([c[i] for i in N])

    A_N = transpose( array([A[:,i] for i in N]) )
    A_B = transpose( array([A[:,i] for i in B]) )
    
    lambda_ = transpose( solve_linear_system(transpose(A_B), c_B) )
    r_N = transpose( dot(lambda_, A_N) - transpose(c_N) )
    

    if max(r_N) <= 0 :
        
        bfs = solve_linear_system(A_B, b)
        x = zeros(A.shape[1])
        
        for i in range(len(B)):
            x[B[i]] = bfs[i]
        
        x = array(x)
             
        A_B_inv = inv(A_B)
        betas = get_betas(x, A_B_inv, len_b, len_c)
        
        H = dot(A_B_inv, A_N)
        gammas = get_gammas(N, r_N, H, len_c, len_b )
        
        sensinfo =  { 
                      'betas' : betas,
                      'gammas' : gammas,
                      'lambda_': lambda_
                    }
        
        ans =  {
                'Status' : 'Solution found', 
                'Optimal Value' : dot(lambda_, b),
                'BFS' : x, 
                'N' : N, 
                'B' : B,
                'sensinfo' : namedtuple('Struct', sensinfo.keys())(*sensinfo.values())
                }
        
        return ans
    
    input_variable = - 1
    output_variable = -1

    pos_first_positive = np.where(r_N > 0)[0][0]
    input_variable = N[pos_first_positive]

    h = solve_linear_system(A_B, b)
    H_e = solve_linear_system(A_B, A[:,input_variable])
    
    if max(H_e) <= 0:
        
        b = transpose(array(b)[np.newaxis])
        aux = solve_linear_system(A_B,b)
        
        size = len(N) + len(B)
        direction = np.zeros(size)
        
        for i in range(size):
            if i in N:
                if i == input_variable:
                    direction[i] = 1
            if i in B:
                direction[i] = H_e[B.index(i)]
         
        position = np.zeros(size)
        for i in range(size):
            if i in B:
                position[i] = aux[B.index(i)]
            
        position = transpose(array(position)[np.newaxis])
        direction = transpose(array(direction)[np.newaxis])
        
        ans =   {
                'Status' : 'Not bounded problem',
                'Descent direction': {'position': position, 'direction': direction}
                }
        return ans

    pos_min_quotient = min ( 
                            enumerate(h), 
                            key=lambda x: x[1]/H_e[x[0]] if H_e[x[0]] > 0 else np.inf
                           )[0]
    output_variable = B[pos_min_quotient]

    pos_output_variable = np.where(B == output_variable)[0][0]
    B[pos_output_variable] = input_variable
    
    pos_input_variable = np.where(N == input_variable)[0][0]
    N[pos_input_variable] = output_variable
    
    ans =   {
            'Status' : 'Method continues',
            'B' : B,
            'N' : N
            }

    return ans


