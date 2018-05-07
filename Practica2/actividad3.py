import simplex_functions as sp
import numpy as np
from scipy.optimize import linprog
import random
from numpy import hstack, array, eye, zeros, arange, transpose, dot, inf

def main():
    A = array([[6.,4.],[8.,4.],[3.,3.]])
    A = transpose(A)

    Nombres = ['David', 'Diana', 'Lidia']

    horas = {}
    horas['David']=[40.,35.,37.,39.,41.,43.,45.]
    horas['Diana']=[40.,35.,37.,39.,41.,43.,45.]
    horas['Lidia']=[20.,15.,17.,19.,21.,23.,25.]

    c=np.array([300.,200.])

    print('Nota: la cantidad de relojes óptima es lamo.\n')
    print('Original:')

    aux_arr = array([horas['David'][0],horas['Diana'][0],horas['Lidia'][0]])
    r = sp.mSimplexDual(A,c,aux_arr)
    print("\n".join("{}:\t{}".format(k, v) for k, v in r.items()))

    x_0 = r['x_0']
    lamo = r['lamo']

    print('Comprobación: x_0*b, lambda*c')
    print( dot(x_0, aux_arr), dot(lamo,c) )


    for nombre in Nombres:
        print('\n\nCambiando a ' + nombre +':')
        for i in horas[nombre][1:]:

            print('\n' + str(i),'horas')
            
            aux_arr =[]
            for name in Nombres:
                if name == nombre:
                    aux_arr.append(i)
                else:
                    aux_arr.append(horas[name][0])
                    
            aux_arr =  array(aux_arr)
            r = sp.mSimplexDual( A, c , aux_arr)
            x_0 = r['x_0']
            lamo = r['lamo']

            print("\n".join("{}:\t{}".format(k, v) for k, v in r.items()))
            
            print('Comprobación:')
            print('x_0*b:', dot(x_0,aux_arr))
            print('lambda*c', dot(lamo,c))


if __name__ == '__main__':
    main()