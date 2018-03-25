# -*- coding: utf-8 -*-
import numpy as np
from numpy import ones, zeros, array

def generaKleeMinty(m):
    c =- ones(m)
    A = zeros((m,m),dtype=float)
    b = array([(2**(i+1))-1 for i in range(m)],dtype=float)
    
    for i in range(m):
        A[i,i]=1
        for j in range(i):
            A[i,j]=2
            
    return(A,b,c)
