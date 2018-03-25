# -*- coding: utf-8 -*-

import numpy as np
from numpy import hstack, array, eye, zeros, arange, transpose, dot
import scipy 
from scipy.linalg import inv


def simplex_step(A,b,c,B,N):

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

            Status          :
            Optimal Value   :
            BFS             :
            N               :
            B               :

    """

    c_B = array([c[i] for i in B])
    c_N = array([c[i] for i in N])

    A_N = transpose( array([A[:,i] for i in N]) )
    A_B = transpose( array([A[:,i] for i in B]) )

    A_B_inv = inv(A_B)
    lambda_ = dot(transpose(c_B), A_B_inv)
    
    r_N = transpose( dot(lambda_, A_N) - transpose(c_N) )
    
    if max(r_N) <= 0 :

        ans =  {'Status' : 'Solution found', 
                'Optimal Value' : dot(lambda_, b),
                'BFS' : dot(A_B_inv, b), 
                'N' : N, 
                'B' : B 
                }

        return ans
    
    input_variable =- 1
    output_variable = -1

    pos_first_positive = np.where(r_N > 0)[0][0]
    input_variable = N[pos_first_positive]

    h = dot(A_B_inv, b)
    H_e = dot(A_B_inv, A[:,input_variable])
    
    if max(H_e) <= 0:

        ans =   {
                'Status' : 'Not bounded problem'
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


def solve(A,b,c, max_iter = 100000):

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

            x0:     Basic feasible solution of the optimization problem 
            z0:     Optimum value of the optimization problem 
            ban:    Several cases:
                -1  Feasible set is empty
                0   Optimal value found
                1   Non-bounded problem
            iter:   Number of iterations

    """

    length = len(b)

    A = hstack( (A, eye(length)) )
    c = hstack( (c, zeros(length)) )
     
    N = np.arange(0, A.shape[1]-length)
    B = np.arange(A.shape[1]-length, A.shape[1])
    
    num_iter = 0
    while num_iter < max_iter:
        ans = simplex_step(A, b, c, B, N)
        
        if ans['Status'] == "Solution found":

            x = zeros(A.shape[1])

            for i in range(len(B)):
                x[B[i]] = ans['BFS'][i]

            ans =   {
                    'x0': x[0:length],
                    'z0': ans['Optimal Value'], 
                    'ban': 0, 
                    'iter': num_iter 
                    }
            return ans

        elif ans['Status'] == 'Not bounded problem':

            ans =   {
                    'x0': None,
                    'z0': None, 
                    'ban': 1, 
                    'iter': num_iter 
                    }
            
            return ans
 
        num_iter += 1

        B = ans['B']
        N = ans['N']
    
    return 'Número máximo de iteraciones alcanzado'

