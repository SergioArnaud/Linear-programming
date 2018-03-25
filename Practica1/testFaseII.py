import numpy as np
from numpy import array
from mSimplexFaseII import solve
from scipy.optimize import linprog
import pprint
from math import log, exp
from numpy.random import rand, normal
from numpy import round, int, abs, array, transpose


def main():

    #Primer test
    A = array([[1,0], [0, 2], [3, 2]])
    b = [4, 12, 18]
    c = array([-3, -5])

    print('\n - - - - - - - - - - - \n')
    print('TEST 1:\n')
    print('Our solution:')
    r = solve(A,b,c)
    print("\n".join("{}:\t{}".format(k, v) for k, v in r.items()))
    print('\nPython solution:')
    print(linprog(c, A_ub=A, b_ub=b))
    print('\n - - - - - - - - - - - \n')


    #Segundo test
    A = array([[-1, 1], [1, 0]])
    b = [0, 2]
    c = array([0, -1])

    print('TEST 2:\n')
    r = solve(A,b,c)
    print('Our solution:')
    print("\n".join("{}:\t{}".format(k, v) for k, v in r.items()))
    print('\nPython solution:')
    print(linprog(c, A_ub=A, b_ub=b))


    #Random tests
    num_random_tests = 5
    eps = 1e-6
    k = 1
    for i in range(5):
        print('\n - - - - - - - - - - - \n')
        print('RANDOM TEST ', k,': ')
        k += 1

        m = int(round(10*exp(log(20)*rand())))
        n = int(round(10*exp(log(20)*rand())))
        sigma = 100

        A = round(sigma*normal(0,1,(n,n)))

        b = round(sigma*abs(normal(0,1,(n,1))))
        b = b[:,0]

        c = round(sigma*normal(0,1,(n,1)))
        c = c[:,0]
        
        our_ans = solve(A,b,c)
        python_ans = linprog(c, A_ub=A, b_ub=b)
        

        if our_ans['x0'] is None:
            if 'The problem appears to be unbounded' in python_ans['message'] and our_ans['ban'] == 1:
                print('Successfull test!')
            else:
                print('Something went wrong')

            continue
        
        if abs(python_ans['fun'] - our_ans['z0']) > eps:
            print('Something went wrong')
            continue
            
        print('Successfull test!')

if __name__ == '__main__':
    main()