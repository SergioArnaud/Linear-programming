from generaKleeMinty import generaKleeMinty
import time
from mSimplexFaseII import solve
import numpy as np

def SimplexKleeMinty(m):
    A,b,c=generaKleeMinty(m)
    
    inicio=time.time()
    resultado=solve(A,b,c)
    final=time.time()
    
    return(m,resultado['iter'],final-inicio)



def main():
    print('m, it,  tiempo ejecucion(segs) ')
    for i in range(3,11):
        print(SimplexKleeMinty(i))

if __name__ == '__main__':
    main()