import sympy

def initialize_parameters(nrateconstants,nvariables):

    mumat = sympy.Matrix(nvariables, 1, lambda i,j:sympy.var('y_%d' % (i)))
    Mumat = sympy.Matrix(nvariables, 1, lambda i,j:sympy.var('mu_%d' % (i)))
    c = sympy.Matrix(nrateconstants, 1, lambda i,j:sympy.var('c_%d' % (i)))

    return [mumat, Mumat, c]
