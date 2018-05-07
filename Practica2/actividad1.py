import simplex_functions as sp
import numpy as np
from scipy.optimize import linprog
import random
from numpy import round, int, abs, array, transpose
from numpy import hstack, array, eye, zeros, arange, transpose, dot, inf
from scipy.linalg import solve as solve_linear_system



def main():

    # Pregunta 1


    A = array([[6, 4], [8, 4], [3,3]])
    b = array([40,40,20])
    c = array([300,200])
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('\n\nPregunta 1: Formular el problema de programación lineal: (maximización)')
    print('A:', A)
    print('b:', b)
    print('c:', c)

    print('\nY la solución a dicho problema esta dada por:\n')
    r = sp.mSimplexMax(A,b,c)
    print("\n".join("{}:\t{}".format(k, v) for k, v in r.items()))

    A_B = array([[6, 4, 1],
         [8, 4, 0],
         [3, 3, 0]])

    x = transpose(array([[10/3, 10/3]]))

    # Pregunta 3
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('\n\nPregunta 3:')

    print('Si cambiamos el precio de el pedestal a 375:')
    c_nuevo = array([375,200])
    r = sp.mSimplexMax(A,b,c_nuevo)

    print('x0:\n', r['x0'])
    print('z0:\n', r['z0'])

    print('Si además el precio de el de pared a 175:')
    c_nuevo = array([300,175])
    r = sp.mSimplexMax(A,b,c_nuevo)

    print('x0:\n', r['x0'])
    print('z0:\n', r['z0'])

    print('Y efectivamente, era de esperarse que el valor donde se obtiene el máximo no\n cambiara porque dichas cantidades están en los intervalos de las gammas')


    #Pregunta 4
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('\n\nPregunta 4:')

    r = sp.mSimplexMax(A,b,c)
    sensinfo = r['sensinfo']
    print('Los relojes de pedestal pueden tener un cambio en el rango:', sensinfo.gammas[0])
    print('Los relojes de pared pueden tener un cambio en el rango:', sensinfo.gammas[1])

    #Pregunta 5
    c_aux = array([[300,200]])

    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('\n\nPregunta 5:')

    b_nuevo = array([45,45,25])
    r = sp.mSimplexMax(A,b_nuevo,c)
    print('Nueva solución óptima (realizando simplex con los nuevos valores):\n', r['x0'])
    print('Nuevo valor máximo:\n', r['z0'])

    print('Ahora usando el hecho de que la base ópptima no cambia: \n(cambios estan en los intervalos de las beta)')
    betas = transpose(array([[5,5,5]]))
    aux = solve_linear_system(A_B,betas)[:2]
    x_n = x + aux
    print('Solución óptima:\n', transpose(x_n)[0])
    print('Ganancia:\n', dot(c_aux,aux)[0][0] )


    #Pregunta 6
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('\n\nPregunta 6:')

    r = sp.mSimplexMax(A,b,c)
    ans = r['x0'][np.newaxis]
    print('Ax = ', dot(A,transpose(ans)))

    print('David puede reducir sus horas de trabajo hasta 33.333\nsin afectar los valores óptimos:')

    b_nuevo = array([33.3333,40,20])
    r = sp.mSimplexMax(A,b_nuevo,c)
    print('x0:\n', r['x0'])
    print('z0:\n', r['z0'])

    print('Sin embargo, si se reduce un poco mas, digamos a 33.30\nya se ven afectados dichos valores')
    b_nuevo = array([33.3,40,20])
    r = sp.mSimplexMax(A,b_nuevo,c)
    print('x0:\n', r['x0'])
    print('z0:\n', r['z0'])

    #Pregunta 7
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('\n\nPregunta 7:')

    Nombres = ['David', 'Diana', 'Lidia']
    cambios = [-5 + (2*i) for i in range(6)]

    A = array([[6, 4], [8, 4], [3,3]])
    b = array([40,40,20])
    c = array([300,200])

    r = sp.mSimplexMax(A,b,c)

    for i,nombre in enumerate(Nombres):
        print('Considerando cambios en', nombre +':')
        for cambio in cambios:
            
            b_nuevo = array(b)
            b_nuevo[i] += cambio
            print('Con b^T = ', b_nuevo)

            r = sp.mSimplexMax(A,b_nuevo,c)            
            print('Solución óptima (Realizando simplex con los nuevos valores)\n', r['x0'])
            print('Valor óptimo\n', r['z0'])

            betas = transpose(array([[0,0,0]]))
            betas[i] = cambio
            aux = solve_linear_system(A_B,betas)[:2]
            x_n = x + aux
            print('Solucion óptima (Usando el hecho de que la base óptima no cambia)\n', transpose(x_n)[0])
            print('Ganancia/Perdida:\n', dot(c_aux,aux)[0][0] )
            

            print()

    #Pregunta 10
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('\n\nPregunta 10:')

    betas = transpose(array([[0,0,5]]))
    aux = solve_linear_system(A_B,betas)[:2]
    x_n = x + aux

    print('Solución óptima:\n', transpose(x_n)[0])
    print('Ganancia:\n', dot(c_aux,aux)[0][0] )




if __name__ == '__main__':
    main()