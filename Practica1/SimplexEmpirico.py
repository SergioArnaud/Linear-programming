from math import log
from math import exp
import numpy as np
from numpy.random import rand
from numpy.random import normal
from numpy import round, int, abs, array, transpose
from scipy.optimize import linprog
from mSimplexFaseII import solve

from bokeh.plotting import figure, show, output_file


def test_simplex(num_tests = 50):
    plot_params = []

    for i in range(num_tests):
        
        m = int(round(10*exp(log(20)*rand())))
        n = int(round(10*exp(log(20)*rand())))
        sigma = 100
        
        A = round(sigma*normal(0,1,(m,n)))
        
        b = round(sigma*abs(normal(0,1,(m,1))))
        b = b[:,0]
        
        c = round(sigma*normal(0,1,(n,1)))
        c = c[:,0]
        
        res = solve(A,b,c)
                            
        iteraciones = res['iter']
        estatus = res['ban']
        
        plot_params.append([n,m,iteraciones, estatus])

    return plot_params


def fit_line(min_n_m, it):

    fit = np.polyfit(np.log10(np.array(min_n_m)), np.log10(np.array(it)), 1)

    m = fit[0]
    b = fit[1]

    x = np.linspace(min(min_n_m), max(min_n_m),100)

    y = m*np.log10(x) + b 
    y = np.power(10,y)

    return m, b, x, y


def make_plot(x_bounded, y_bounded, x_unbounded, y_unbounded, x, y):
    p = figure(title="Ejercicio 3", y_axis_type="log", x_axis_type='log')

    p.circle(x_bounded, y_bounded, color='blue', legend='bounded')
    p.circle(x_unbounded, y_unbounded, color='red', legend='unbounded')
    p.line(x, y, color='green', line_width=1.5)

    p.legend.location = "top_left"

    p.xaxis.axis_label = 'min {n,m}'
    p.yaxis.axis_label = 'Número de iteraciones'

    output_file("logplot.html", title="log plot example")

    show(p)


def main():

    plot_params = test_simplex(50)

    n,m,it,status = zip(*plot_params)
    min_n_m = [min(a[0] , a[1]) for a in zip(n,m)]
    
    x_bounded = [a[0] for a in zip(min_n_m,status) if a[1] == 0 ]
    y_bounded = [a[0] for a in zip(it,status) if a[1] == 0 ]
    x_unbounded = [a[0] for a in zip(min_n_m,status) if a[1] == 1 ]
    y_unbounded = [a[0] for a in zip(it,status) if a[1] == 1 ]

    m, b, x, y = fit_line(min_n_m, it)

    print('Pendiente de la recta:')
    print(m)
    print('Ordenada al origen de la recta:')
    print(b)

    make_plot(x_bounded, y_bounded, x_unbounded, y_unbounded, x, y)



if __name__ == "__main__":
    main()



