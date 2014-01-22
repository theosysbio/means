import unittest
import sympy
from centralmoments import eq_centralmoments
from sympy import Matrix, Symbol
from math import factorial
from initialize_parameters import initialize_parameters
from sympy import sympify

# eq_centralmoments(counter,mcounter,M,TaylorM,nvariables,ymat,nreactions,nMoments,amat,S,nDerivatives):

class CentralMomentsTestCase(unittest.TestCase):
    def test_centralmoments_using_p53model(self):

        nreactions = 6
        nrateconstants = 7
        nvariables = 3
        ymat = [sympy.var('y_0'), sympy.var('y_1'), sympy.var('y_2')]
        [something, Mumat, c]=initialize_parameters(nrateconstants,nvariables)
        nMoments = 2
        S = Matrix(3,6,[1,-1,-1,0,0,0,0,0,0,1,-1,0,0,0,0,0,1,-1])
        a = Matrix(nreactions, 1, lambda i,j:0)
        a[0] = c[0]
        a[1] = c[1]*ymat[0]
        a[2] = c[2]*ymat[2]*ymat[0]/(ymat[0]+c[6])
        a[3] = c[3]*ymat[0]
        a[4] = c[4]*ymat[1]
        a[5] = c[5]*ymat[2]

        #counter and mcount are obtained from fcount(2,3)
        counter = [[0, 0, 0], [0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]
        mcounter =  [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]

        ymat = [sympy.var('y_0'), sympy.var('y_1'), sympy.var('y_2')]
        nDerivatives = nMoments
        amat = a
        damat = Matrix(nDerivatives,1, lambda i,j:0)

        M = sympify.matrix[[sympify("c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)"),sympify("0") ,sympify("0") ,sympify("0"), sympify("c_2*y_0/(c_6 + y_0)**2 - c_2/(c_6 + y_0)"),sympify("0"), sympify("-c_2*y_0*y_2/(c_6 + y_0)**3 + c_2*y_2/(c_6 + y_0)**2")],
[sympify("c*y_0 - c_4*y_1"), sympify("0"), sympify("0"), sympify("0"), sympify("0"), sympify("0"), sympify("0")]
[sympify(c_4*y_1 - c_5*y_2), sympify("0"), sympify("0"), sympify("0"), sympify("0"), sympify("0"), sympify("0")]

        TaylorM = 0 # not used in centralmoments.py

        centralmoments_test = eq_centralmoments(counter,mcounter,M,TaylorM,nvariables,ymat,nreactions,nMoments,amat,S,nDerivatives)
     #   centralmoments_expected =
        self.assertEqual(len(centralmoments_test), nreactions)
     #   self.assertEqual()




