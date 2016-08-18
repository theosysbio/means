from __future__ import absolute_import, print_function

import unittest

import sympy
from numpy import array

from means.approximation.mea.moment_expansion_approximation import MomentExpansionApproximation
from means.core import Moment, ODEProblem, Model
from means.util.sympyhelpers import sympy_expressions_equal
from means.util.sympyhelpers import to_sympy_matrix

M_2 = sympy.Symbol("M_2")
M_3 = sympy.Symbol("M_3")
M_4 = sympy.Symbol("M_4")
M_0_0_2 = sympy.Symbol("M_0_0_2")
M_0_1_1 = sympy.Symbol("M_0_1_1")
M_0_2_0 = sympy.Symbol("M_0_2_0")
M_1_0_1 = sympy.Symbol("M_1_0_1")
M_1_1_0 = sympy.Symbol("M_1_1_0")
M_2_0_0 = sympy.Symbol("M_2_0_0")
M_0_0_3 = sympy.Symbol("M_0_0_3")
M_0_1_2 = sympy.Symbol("M_0_1_2")
M_0_2_1 = sympy.Symbol("M_0_2_1")
M_0_3_0 = sympy.Symbol("M_0_3_0")
M_1_0_2 = sympy.Symbol("M_1_0_2")
M_1_1_1 = sympy.Symbol("M_1_1_1")
M_1_2_0 = sympy.Symbol("M_1_2_0")
M_2_0_1 = sympy.Symbol("M_2_0_1")
M_2_1_0 = sympy.Symbol("M_2_1_0")
M_3_0_0 = sympy.Symbol("M_3_0_0")
M_0_0_4 = sympy.Symbol("M_0_0_4")
M_0_1_3 = sympy.Symbol("M_0_1_3")
M_0_2_2 = sympy.Symbol("M_0_2_2")
M_0_3_1 = sympy.Symbol("M_0_3_1")
M_0_4_0 = sympy.Symbol("M_0_4_0")
M_1_0_3 = sympy.Symbol("M_1_0_3")
M_1_1_2 = sympy.Symbol("M_1_1_2")
M_1_2_1 = sympy.Symbol("M_1_2_1")
M_1_3_0 = sympy.Symbol("M_1_3_0")
M_2_0_2 = sympy.Symbol("M_2_0_2")
M_2_1_1 = sympy.Symbol("M_2_1_1")
M_2_2_0 = sympy.Symbol("M_2_2_0")
M_3_0_1 = sympy.Symbol("M_3_0_1")
M_3_1_0 = sympy.Symbol("M_3_1_0")
M_4_0_0 = sympy.Symbol("M_4_0_0")
M_0_2 = sympy.Symbol("M_0_2")
M_1_1 = sympy.Symbol("M_1_1")
M_2_0 = sympy.Symbol("M_2_0")
M_0_3 = sympy.Symbol("M_0_3")
M_1_2 = sympy.Symbol("M_1_2")
M_2_1 = sympy.Symbol("M_2_1")
M_3_0 = sympy.Symbol("M_3_0")
M_0_4 = sympy.Symbol("M_0_4")
M_1_3 = sympy.Symbol("M_1_3")
M_2_2 = sympy.Symbol("M_2_2")
M_3_1 = sympy.Symbol("M_3_1")
M_4_0 = sympy.Symbol("M_4_0")

c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7 = sympy.Symbol('c_0'), sympy.Symbol('c_1'), sympy.Symbol('c_2'),\
                                         sympy.Symbol('c_3'), sympy.Symbol('c_4'), sympy.Symbol('c_5'),\
                                         sympy.Symbol('c_6'), sympy.Symbol('c_7')

y_0, y_1, y_2 = sympy.Symbol('y_0'), sympy.Symbol('y_1'), sympy.Symbol('y_2')

class TestMomentExpansionApproximationOnMaxOrderIsThree(unittest.TestCase):
    max_order = 3
    def test_run_dimer(self):
        dimer_model = Model(parameters=['c_0', 'c_1', 'c_2'],
                           species=['y_0'],
                           stoichiometry_matrix=[[-2, 2]],
                           propensities=['c_0*y_0*(y_0-1)',
                                         'c_1*((1.0/2)*(c_2-y_0))'])
        expected_lhs = [
        Moment(array([1]), symbol=y_0),
        Moment(array([2]), symbol=M_2),
        Moment(array([3]), symbol=M_3),
        ]

        expected_mfk = sympy.Matrix([
        [-2*c_0*y_0*(y_0 - 1) - 2*c_0*M_2 + 2*c_1*(0.5*c_2 - 0.5*y_0)],
        [-4*c_0*y_0**2*(y_0 - 1) + 4*c_0*y_0*(y_0 - 1) - 4*c_0*M_3 + 4*c_1*y_0*(0.5*c_2 - 0.5*y_0) + 4*c_1*(0.5*c_2 - 0.5*y_0) - 2*y_0*(-2*c_0*y_0*(y_0 - 1) + 2*c_1*(0.5*c_2 - 0.5*y_0)) + M_2*(4*c_0*y_0 - 4*c_0*(3*y_0 - 1) + 4*c_0 - 2.0*c_1)],
        [-6*c_0*y_0**3*(y_0 - 1) + 12*c_0*y_0**2*(y_0 - 1) - 8*c_0*y_0*(y_0 - 1) + 6*c_1*y_0**2*(0.5*c_2 - 0.5*y_0) + 12*c_1*y_0*(0.5*c_2 - 0.5*y_0) + 8*c_1*(0.5*c_2 - 0.5*y_0) + 6*y_0**2*(-2*c_0*y_0*(y_0 - 1) + 2*c_1*(0.5*c_2 - 0.5*y_0)) - 3*y_0*(-4*c_0*y_0**2*(y_0 - 1) + 4*c_0*y_0*(y_0 - 1) + 4*c_1*y_0*(0.5*c_2 - 0.5*y_0) + 4*c_1*(0.5*c_2 - 0.5*y_0)) + M_2*(-12*c_0*y_0**2 - 18*c_0*y_0*(2*y_0 - 1) + 12*c_0*(3*y_0 - 1) + 6*c_0*(y_0**2 + M_2) - 8*c_0 + 3*c_1*(c_2 - 3.0*y_0) - 6.0*c_1 - 3*y_0*(-4*c_0*(3*y_0 - 1) + 4*c_0 - 2.0*c_1)) + M_3*(12*c_0*y_0 - 6*c_0*(4*y_0 - 1) + 12*c_0 - 3.0*c_1) - 3*(y_0**2 + M_2)*(-2*c_0*y_0*(y_0 - 1) + 2*c_1*(0.5*c_2 - 0.5*y_0))],
        ])

        expected_constants = sympy.Matrix([
        [c_0],
        [c_1],
        [c_2]])


        expected = ODEProblem("MEA", expected_lhs, expected_mfk, expected_constants)
        mea = MomentExpansionApproximation(dimer_model, max_order=self.max_order)
        answer = mea.run()

        self.assertEqual(answer, expected)

    def test_run_mm(self):
        mm_model = Model(parameters=['c_0', 'c_1', 'c_2'],
                               species=['y_0', 'y_1'],
                               propensities=['c_0*y_0*(120-301+y_0+y_1)',
                                             'c_1*(301-(y_0+y_1))',
                                             'c_2*(301-(y_0+y_1))'],
                               stoichiometry_matrix=[[-1, 1, 0],
                                                     [0, 0, 1]])

        expected_lhs = [
        Moment(array([1, 0]), symbol=y_0),
        Moment(array([0, 1]), symbol=y_1),
        Moment(array([0, 2]), symbol=M_0_2),
        Moment(array([1, 1]), symbol=M_1_1),
        Moment(array([2, 0]), symbol=M_2_0),
        Moment(array([0, 3]), symbol=M_0_3),
        Moment(array([1, 2]), symbol=M_1_2),
        Moment(array([2, 1]), symbol=M_2_1),
        Moment(array([3, 0]), symbol=M_3_0),
        ]

        expected_mfk = sympy.Matrix([
        [-c_0*y_0*(y_0 + y_1 - 181) - c_0*M_1_1 - c_0*M_2_0 + c_1*(-y_0 - y_1 + 301)],
        [c_2*(-y_0 - y_1 + 301)],
        [-2*c_2*M_0_2 - 2*c_2*M_1_1 + c_2*(-y_0 - y_1 + 301)],
        [-c_0*y_0*y_1*(y_0 + y_1 - 181) - c_0*M_1_2 - c_0*M_2_1 + c_1*y_1*(-y_0 - y_1 + 301) - c_2*M_2_0 - y_1*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301)) + M_0_2*(-c_0*y_0 - c_1) + M_1_1*(c_0*y_1 - c_0*(2*y_0 + 2*y_1 - 181) - c_1 - c_2)],
        [-2*c_0*y_0**2*(y_0 + y_1 - 181) + c_0*y_0*(y_0 + y_1 - 181) - 2*c_0*M_2_1 - 2*c_0*M_3_0 + 2*c_1*y_0*(-y_0 - y_1 + 301) + c_1*(-y_0 - y_1 + 301) - 2*y_0*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301)) + M_1_1*(-2*c_0*y_0 + c_0 - 2*c_1) + M_2_0*(2*c_0*y_0 - 2*c_0*(3*y_0 + y_1 - 181) + c_0 - 2*c_1)],
        [9*c_2*y_1**2*(-y_0 - y_1 + 301) + 3*c_2*y_1*(-y_0 - y_1 + 301) - 3*c_2*M_1_1 - 3*c_2*M_0_3 - 3*c_2*M_1_2 - 3*c_2*(y_1**2 + M_0_2)*(-y_0 - y_1 + 301) + c_2*(-y_0 - y_1 + 301) - 3*y_1*(2*c_2*y_1*(-y_0 - y_1 + 301) + c_2*(-y_0 - y_1 + 301)) + M_0_2*(6*c_2*y_1 - 3*c_2*(y_0 + 3*y_1 - 301) - 3*c_2)],
        [-c_0*y_0*y_1**2*(y_0 + y_1 - 181) + c_1*y_1**2*(-y_0 - y_1 + 301) + 4*c_2*y_0*y_1*(-y_0 - y_1 + 301) + c_2*y_0*(-y_0 - y_1 + 301) - 2*c_2*M_2_1 - 2*c_2*(y_0*y_1 + M_1_1)*(-y_0 - y_1 + 301) - y_0*(2*c_2*y_1*(-y_0 - y_1 + 301) + c_2*(-y_0 - y_1 + 301)) + 2*y_1*(c_2*y_0*(-y_0 - y_1 + 301) + y_1*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301))) - 2*y_1*(-c_0*y_0*y_1*(y_0 + y_1 - 181) + c_1*y_1*(-y_0 - y_1 + 301) + c_2*y_0*(-y_0 - y_1 + 301)) + M_0_2*(-c_0*y_0*(y_0 + 3*y_1 - 181) - c_1*(y_0 + 3*y_1 - 301) - 2*y_1*(-c_0*y_0 - c_1)) + M_1_1*(-2*c_0*y_1**2 - c_0*y_1*(4*y_0 + 3*y_1 - 362) + c_0*(y_1**2 + M_0_2) - 2*c_1*y_1 + 2*c_2*y_0 - 2*c_2*(2*y_0 + 2*y_1 - 301) - c_2 - 2*y_1*(-c_0*(2*y_0 + 2*y_1 - 181) - c_1 - c_2)) + M_2_0*(-3*c_0*y_1**2 + c_0*(y_1**2 + M_0_2) - 2*c_2*y_1 - c_2 - 2*y_1*(-c_0*y_1 - c_2)) + M_0_3*(-c_0*y_0 - c_1) + M_1_2*(2*c_0*y_1 - c_0*(2*y_0 + 3*y_1 - 181) - c_1 - 2*c_2) - (y_1**2 + M_0_2)*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301))],
        [-2*c_0*y_0**2*y_1*(y_0 + y_1 - 181) + c_0*y_0*y_1*(y_0 + y_1 - 181) + 2*c_1*y_0*y_1*(-y_0 - y_1 + 301) + c_1*y_1*(-y_0 - y_1 + 301) + c_2*y_0**2*(-y_0 - y_1 + 301) - c_2*M_3_0 - c_2*(y_0**2 + M_2_0)*(-y_0 - y_1 + 301) + 2*y_0*y_1*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301)) + 2*y_0*(c_2*y_0*(-y_0 - y_1 + 301) + y_1*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301))) - 2*y_0*(-c_0*y_0*y_1*(y_0 + y_1 - 181) + c_1*y_1*(-y_0 - y_1 + 301) + c_2*y_0*(-y_0 - y_1 + 301)) - y_1*(-2*c_0*y_0**2*(y_0 + y_1 - 181) + c_0*y_0*(y_0 + y_1 - 181) + 2*c_1*y_0*(-y_0 - y_1 + 301) + c_1*(-y_0 - y_1 + 301)) + M_0_2*(-2*c_0*y_0**2 + c_0*y_0 - 2*c_1*y_0 - c_1 - 2*y_0*(-c_0*y_0 - c_1)) + M_1_1*(-4*c_0*y_0*y_1 - 2*c_0*y_0*(3*y_0 + 4*y_1 - 362) + 2*c_0*(y_0*y_1 + M_1_1) + c_0*(2*y_0 + 2*y_1 - 181) - 2*c_1*(2*y_0 + 2*y_1 - 301) - c_1 - 2*c_2*y_0 - 2*y_0*(-c_0*(2*y_0 + 2*y_1 - 181) - c_1 - c_2) - y_1*(-4*c_0*y_0 + c_0 - 2*c_1)) + M_2_0*(-4*c_0*y_0*y_1 - 2*c_0*y_1*(3*y_0 + y_1 - 181) + c_0*y_1 + 2*c_0*(y_0*y_1 + M_1_1) - 2*c_1*y_1 - c_2*(3*y_0 + y_1 - 301) - 2*y_0*(-c_0*y_1 - c_2) - y_1*(-2*c_0*(3*y_0 + y_1 - 181) + c_0 - 2*c_1)) + M_1_2*(-2*c_0*y_0 + c_0 - 2*c_1) + M_2_1*(2*c_0*y_0 + 2*c_0*y_1 - 2*c_0*(3*y_0 + 2*y_1 - 181) + c_0 - 2*c_1 - c_2) - 2*(y_0*y_1 + M_1_1)*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301))],
        [-3*c_0*y_0**3*(y_0 + y_1 - 181) + 3*c_0*y_0**2*(y_0 + y_1 - 181) - c_0*y_0*(y_0 + y_1 - 181) + 3*c_1*y_0**2*(-y_0 - y_1 + 301) + 3*c_1*y_0*(-y_0 - y_1 + 301) + c_1*(-y_0 - y_1 + 301) + 6*y_0**2*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301)) - 3*y_0*(-2*c_0*y_0**2*(y_0 + y_1 - 181) + c_0*y_0*(y_0 + y_1 - 181) + 2*c_1*y_0*(-y_0 - y_1 + 301) + c_1*(-y_0 - y_1 + 301)) + M_1_1*(-15*c_0*y_0**2 + 6*c_0*y_0 + 3*c_0*(y_0**2 + M_2_0) - c_0 - 6*c_1*y_0 - 3*c_1 - 3*y_0*(-4*c_0*y_0 + c_0 - 2*c_1)) + M_2_0*(-6*c_0*y_0**2 - 9*c_0*y_0*(2*y_0 + y_1 - 181) + 3*c_0*(y_0**2 + M_2_0) + 3*c_0*(3*y_0 + y_1 - 181) - c_0 - 3*c_1*(3*y_0 + y_1 - 301) - 3*c_1 - 3*y_0*(-2*c_0*(3*y_0 + y_1 - 181) + c_0 - 2*c_1)) + M_2_1*(-3*c_0*y_0 + 3*c_0 - 3*c_1) + M_3_0*(6*c_0*y_0 - 3*c_0*(4*y_0 + y_1 - 181) + 3*c_0 - 3*c_1) - 3*(y_0**2 + M_2_0)*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301))],
        ])

        expected_constants = sympy.Matrix([
        [c_0],
        [c_1],
        [c_2]])



        expected = ODEProblem("MEA", expected_lhs, expected_mfk, expected_constants)
        mea = MomentExpansionApproximation(mm_model, max_order=self.max_order)
        answer = mea.run()

        self.assertEqual(answer, expected)

    def test_run_hes1(self):
        hes1_model = Model(parameters=['c_0', 'c_1', 'c_2', 'c_3'],
                   species=['y_0', 'y_1', 'y_2'],
                   propensities=['0.03*y_0',
                                 '0.03*y_1',
                                 '0.03*y_2',
                                 'c_3*y_1',
                                 'c_2*y_0',
                                 '1.0/(1+(y_2/c_0)**2)'],
                   stoichiometry_matrix=[[-1, 0, 0, 0, 0, 1],
                                         [0, -1, 0, -1, 1, 0],
                                         [0, 0, -1, 1, 0, 0]])
        expected_lhs = [
        Moment(array([1, 0, 0]), symbol=y_0),
        Moment(array([0, 1, 0]), symbol=y_1),
        Moment(array([0, 0, 1]), symbol=y_2),
        Moment(array([0, 0, 2]), symbol=M_0_0_2),
        Moment(array([0, 1, 1]), symbol=M_0_1_1),
        Moment(array([0, 2, 0]), symbol=M_0_2_0),
        Moment(array([1, 0, 1]), symbol=M_1_0_1),
        Moment(array([1, 1, 0]), symbol=M_1_1_0),
        Moment(array([2, 0, 0]), symbol=M_2_0_0),
        Moment(array([0, 0, 3]), symbol=M_0_0_3),
        Moment(array([0, 1, 2]), symbol=M_0_1_2),
        Moment(array([0, 2, 1]), symbol=M_0_2_1),
        Moment(array([0, 3, 0]), symbol=M_0_3_0),
        Moment(array([1, 0, 2]), symbol=M_1_0_2),
        Moment(array([1, 1, 1]), symbol=M_1_1_1),
        Moment(array([1, 2, 0]), symbol=M_1_2_0),
        Moment(array([2, 0, 1]), symbol=M_2_0_1),
        Moment(array([2, 1, 0]), symbol=M_2_1_0),
        Moment(array([3, 0, 0]), symbol=M_3_0_0),
        ]

        expected_mfk = sympy.MutableDenseMatrix([[-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2) + M_0_0_2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) + y_2*M_0_0_3*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(6*c_0**4*(1 + y_2**2/c_0**2)**3)],
        [c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1],
        [c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2],
        [2*c_3*y_1*y_2 + c_3*y_1 + 2*c_3*M_0_1_1 - sympy.Float('0.059999999999999998', prec=15)*y_2**2 - 2*y_2*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) + sympy.Float('0.029999999999999999', prec=15)*y_2 - sympy.Float('0.059999999999999998', prec=15)*M_0_0_2],
        [c_2*y_0*y_2 + c_2*M_1_0_1 + c_3*y_1**2 - c_3*y_1*y_2 - c_3*y_1 + c_3*M_0_2_0 - sympy.Float('0.059999999999999998', prec=15)*y_1*y_2 - y_1*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) - y_2*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) + M_0_1_1*(-c_3 + sympy.Float('-0.059999999999999998', prec=15))],
        [2*c_2*y_0*y_1 + c_2*y_0 + 2*c_2*M_1_1_0 - 2*c_3*y_1**2 + c_3*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_1**2 - 2*y_1*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) + sympy.Float('0.029999999999999999', prec=15)*y_1 + M_0_2_0*(-2*c_3 + sympy.Float('-0.059999999999999998', prec=15))],
        [c_3*y_0*y_1 + c_3*M_1_1_0 - sympy.Float('0.059999999999999998', prec=15)*y_0*y_2 - y_0*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) - y_2*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2)) + sympy.Float('1.0', prec=15)*y_2/(1 + y_2**2/c_0**2) + M_0_0_2*(y_2*(sympy.Float('-6.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) - y_2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2)) - sympy.Float('0.059999999999999998', prec=15)*M_1_0_1 + M_0_0_3*((sympy.Float('-6.0', prec=15) + sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)) - sympy.Float('48.0', prec=15)*y_2**4/(c_0**4*(1 + y_2**2/c_0**2)**2))/(6*c_0**2*(1 + y_2**2/c_0**2)**2) - y_2**2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(6*c_0**4*(1 + y_2**2/c_0**2)**3))],
        [c_2*y_0**2 + c_2*M_2_0_0 - c_3*y_0*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_0*y_1 - y_0*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) - y_1*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2)) + sympy.Float('1.0', prec=15)*y_1/(1 + y_2**2/c_0**2) + M_1_1_0*(-c_3 + sympy.Float('-0.059999999999999998', prec=15)) - sympy.Float('2.0', prec=15)*y_2*M_0_1_1/(c_0**2*(1 + y_2**2/c_0**2)**2) + M_0_1_2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2)],
        [-sympy.Float('0.059999999999999998', prec=15)*y_0**2 - 2*y_0*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2)) + sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('2.0', prec=15)*y_0/(1 + y_2**2/c_0**2) - sympy.Float('0.059999999999999998', prec=15)*M_2_0_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2) - sympy.Float('4.0', prec=15)*y_2*M_1_0_1/(c_0**2*(1 + y_2**2/c_0**2)**2) + M_0_0_2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) + M_1_0_2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(c_0**2*(1 + y_2**2/c_0**2)**2) + y_2*M_0_0_3*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(6*c_0**4*(1 + y_2**2/c_0**2)**3)],
        [3*c_3*y_1*y_2**2 + 3*c_3*y_1*y_2 + c_3*y_1 + 3*c_3*M_0_1_1 + 3*c_3*M_0_1_2 - sympy.Float('0.089999999999999997', prec=15)*y_2**3 + 6*y_2**2*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) + sympy.Float('0.089999999999999997', prec=15)*y_2**2 - 3*y_2*(2*c_3*y_1*y_2 + c_3*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_2**2 + sympy.Float('0.029999999999999999', prec=15)*y_2) - sympy.Float('0.029999999999999999', prec=15)*y_2 + M_0_0_2*(3*c_3*y_1 - sympy.Float('0.090000000000000024', prec=15)*y_2 + sympy.Float('0.089999999999999997', prec=15)) - sympy.Float('0.089999999999999997', prec=15)*M_0_0_3 - 3*(y_2**2 + M_0_0_2)*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2)],
        [c_2*y_0*y_2**2 + c_2*M_1_0_2 + 2*c_3*y_1**2*y_2 + c_3*y_1**2 - c_3*y_1*y_2**2 - 2*c_3*y_1*y_2 - c_3*y_1 + c_3*M_0_2_0 + 2*c_3*M_0_2_1 - sympy.Float('0.089999999999999997', prec=15)*y_1*y_2**2 + 2*y_1*y_2*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) + sympy.Float('0.029999999999999999', prec=15)*y_1*y_2 - y_1*(2*c_3*y_1*y_2 + c_3*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_2**2 + sympy.Float('0.029999999999999999', prec=15)*y_2) + 2*y_2*(y_1*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) + y_2*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1)) - 2*y_2*(c_2*y_0*y_2 + c_3*y_1**2 - c_3*y_1*y_2 - c_3*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_1*y_2) + M_0_0_2*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) + M_0_1_1*(2*c_3*y_1 - 2*c_3*y_2 - 2*c_3 - 2*y_2*(-c_3 + sympy.Float('-0.059999999999999998', prec=15)) - sympy.Float('0.17999999999999999', prec=15)*y_2 + sympy.Float('0.029999999999999999', prec=15)) + M_0_1_2*(-c_3 + sympy.Float('-0.089999999999999997', prec=15)) - (y_2**2 + M_0_0_2)*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) - 2*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2)*(y_1*y_2 + M_0_1_1)],
        [2*c_2*y_0*y_1*y_2 + c_2*y_0*y_2 + 2*c_2*M_1_1_1 + c_2*M_1_0_1 + c_3*y_1**3 - 2*c_3*y_1**2*y_2 - 2*c_3*y_1**2 + c_3*y_1*y_2 + c_3*y_1 + c_3*M_0_3_0 - sympy.Float('0.089999999999999997', prec=15)*y_1**2*y_2 + 2*y_1*y_2*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) + sympy.Float('0.029999999999999999', prec=15)*y_1*y_2 + 2*y_1*(y_1*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) + y_2*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1)) - 2*y_1*(c_2*y_0*y_2 + c_3*y_1**2 - c_3*y_1*y_2 - c_3*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_1*y_2) - y_2*(2*c_2*y_0*y_1 + c_2*y_0 - 2*c_3*y_1**2 + c_3*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_1**2 + sympy.Float('0.029999999999999999', prec=15)*y_1) + M_0_1_1*(2*c_2*y_0 - 4*c_3*y_1 + c_3 - 2*y_1*(-c_3 + sympy.Float('-0.059999999999999998', prec=15)) - sympy.Float('0.17999999999999999', prec=15)*y_1 + sympy.Float('0.029999999999999999', prec=15)) + M_0_2_0*(c_3*y_1 - 2*c_3*y_2 - 2*c_3 - y_2*(-2*c_3 + sympy.Float('-0.059999999999999998', prec=15)) - sympy.Float('0.089999999999999997', prec=15)*y_2) + M_0_2_1*(-2*c_3 + sympy.Float('-0.089999999999999997', prec=15)) - (y_1**2 + M_0_2_0)*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) - 2*(y_1*y_2 + M_0_1_1)*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1)],
        [3*c_2*y_0*y_1**2 + 3*c_2*y_0*y_1 + c_2*y_0 + 3*c_2*M_1_2_0 + 3*c_2*M_1_1_0 - 3*c_3*y_1**3 + 3*c_3*y_1**2 - c_3*y_1 - sympy.Float('0.089999999999999997', prec=15)*y_1**3 + 6*y_1**2*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) + sympy.Float('0.089999999999999997', prec=15)*y_1**2 - 3*y_1*(2*c_2*y_0*y_1 + c_2*y_0 - 2*c_3*y_1**2 + c_3*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_1**2 + sympy.Float('0.029999999999999999', prec=15)*y_1) - sympy.Float('0.029999999999999999', prec=15)*y_1 + M_0_3_0*(-3*c_3 + sympy.Float('-0.089999999999999997', prec=15)) + M_0_2_0*(3*c_2*y_0 - 9*c_3*y_1 + 3*c_3 - 3*y_1*(-2*c_3 + sympy.Float('-0.059999999999999998', prec=15)) - sympy.Float('0.27000000000000002', prec=15)*y_1 + sympy.Float('0.089999999999999997', prec=15)) - 3*(y_1**2 + M_0_2_0)*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1)],
        [2*c_3*y_0*y_1*y_2 + c_3*y_0*y_1 + 2*c_3*M_1_1_1 + c_3*M_1_1_0 - sympy.Float('0.089999999999999997', prec=15)*y_0*y_2**2 + 2*y_0*y_2*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) + sympy.Float('0.029999999999999999', prec=15)*y_0*y_2 - y_0*(2*c_3*y_1*y_2 + c_3*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_2**2 + sympy.Float('0.029999999999999999', prec=15)*y_2) + sympy.Float('1.0', prec=15)*y_2**2/(1 + y_2**2/c_0**2) + 2*y_2*(y_0*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) + y_2*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2))) - 2*y_2*(c_3*y_0*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_0*y_2 + sympy.Float('1.0', prec=15)*y_2/(1 + y_2**2/c_0**2)) + M_0_0_2*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + (sympy.Float('2.0', prec=15) - sympy.Float('10.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)) + sympy.Float('8.0', prec=15)*y_2**4/(c_0**4*(1 + y_2**2/c_0**2)**2))/(2*(1 + y_2**2/c_0**2)) - y_2**2*(sympy.Float('-6.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(c_0**2*(1 + y_2**2/c_0**2)**2) + y_2**2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(c_0**2*(1 + y_2**2/c_0**2)**2) - (sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))*(y_2**2 + M_0_0_2)/(2*c_0**2*(1 + y_2**2/c_0**2)**2)) - sympy.Float('0.089999999999999997', prec=15)*M_1_0_2 + M_1_0_1*(2*c_3*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_2 + sympy.Float('0.029999999999999999', prec=15)) + M_0_0_3*(y_2*(sympy.Float('-24.0', prec=15) + sympy.Float('72.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)) - sympy.Float('48.0', prec=15)*y_2**4/(c_0**4*(1 + y_2**2/c_0**2)**2))/(6*c_0**2*(1 + y_2**2/c_0**2)**2) - y_2*(sympy.Float('-6.0', prec=15) + sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)) - sympy.Float('48.0', prec=15)*y_2**4/(c_0**4*(1 + y_2**2/c_0**2)**2))/(3*c_0**2*(1 + y_2**2/c_0**2)**2) + y_2**3*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(3*c_0**4*(1 + y_2**2/c_0**2)**3) - y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))*(y_2**2 + M_0_0_2)/(6*c_0**4*(1 + y_2**2/c_0**2)**3)) - (-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2))*(y_2**2 + M_0_0_2) - 2*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2)*(y_0*y_2 + M_1_0_1)],
        [c_2*y_0**2*y_2 + c_2*M_2_0_1 + c_3*y_0*y_1**2 - c_3*y_0*y_1*y_2 - c_3*y_0*y_1 + c_3*M_1_2_0 - sympy.Float('0.089999999999999997', prec=15)*y_0*y_1*y_2 + y_0*(y_1*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) + y_2*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1)) - y_0*(c_2*y_0*y_2 + c_3*y_1**2 - c_3*y_1*y_2 - c_3*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_1*y_2) + sympy.Float('1.0', prec=15)*y_1*y_2/(1 + y_2**2/c_0**2) + y_1*(y_0*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) + y_2*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2))) - y_1*(c_3*y_0*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_0*y_2 + sympy.Float('1.0', prec=15)*y_2/(1 + y_2**2/c_0**2)) + y_2*(y_0*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) + y_1*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2))) - y_2*(c_2*y_0**2 - c_3*y_0*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_0*y_1 + sympy.Float('1.0', prec=15)*y_1/(1 + y_2**2/c_0**2)) + M_0_0_2*(y_1*y_2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) - (sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))*(y_1*y_2 + M_0_1_1)/(2*c_0**2*(1 + y_2**2/c_0**2)**2)) + M_1_1_1*(-c_3 + sympy.Float('-0.089999999999999997', prec=15)) + M_0_1_1*(-c_3*y_0 - y_0*(-c_3 + sympy.Float('-0.059999999999999998', prec=15)) - sympy.Float('0.089999999999999997', prec=15)*y_0 + (sympy.Float('1.0', prec=15) - sympy.Float('2.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(1 + y_2**2/c_0**2) + sympy.Float('2.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)**2)) + M_1_0_1*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) + M_1_1_0*(c_3*y_1 - c_3*y_2 - c_3 - y_2*(-c_3 + sympy.Float('-0.059999999999999998', prec=15)) - sympy.Float('0.089999999999999997', prec=15)*y_2) + M_0_0_3*(y_1*y_2**2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(6*c_0**4*(1 + y_2**2/c_0**2)**3) - y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))*(y_1*y_2 + M_0_1_1)/(6*c_0**4*(1 + y_2**2/c_0**2)**3)) + M_0_1_2*(y_2*(sympy.Float('-6.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) - y_2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2)) - (-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2))*(y_1*y_2 + M_0_1_1) - (c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2)*(y_0*y_1 + M_1_1_0) - (y_0*y_2 + M_1_0_1)*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1)],
        [2*c_2*y_0**2*y_1 + c_2*y_0**2 + 2*c_2*M_2_1_0 + c_2*M_2_0_0 - 2*c_3*y_0*y_1**2 + c_3*y_0*y_1 - sympy.Float('0.089999999999999997', prec=15)*y_0*y_1**2 + 2*y_0*y_1*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) + sympy.Float('0.029999999999999999', prec=15)*y_0*y_1 - y_0*(2*c_2*y_0*y_1 + c_2*y_0 - 2*c_3*y_1**2 + c_3*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_1**2 + sympy.Float('0.029999999999999999', prec=15)*y_1) + sympy.Float('1.0', prec=15)*y_1**2/(1 + y_2**2/c_0**2) + 2*y_1*(y_0*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) + y_1*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2))) - 2*y_1*(c_2*y_0**2 - c_3*y_0*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_0*y_1 + sympy.Float('1.0', prec=15)*y_1/(1 + y_2**2/c_0**2)) + M_0_0_2*(y_1**2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) - (sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))*(y_1**2 + M_0_2_0)/(2*c_0**2*(1 + y_2**2/c_0**2)**2)) + M_1_2_0*(-2*c_3 + sympy.Float('-0.089999999999999997', prec=15)) + M_0_2_0*(-2*c_3*y_0 - y_0*(-2*c_3 + sympy.Float('-0.059999999999999998', prec=15)) - sympy.Float('0.089999999999999997', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2)) + M_1_1_0*(2*c_2*y_0 - 4*c_3*y_1 + c_3 - 2*y_1*(-c_3 + sympy.Float('-0.059999999999999998', prec=15)) - sympy.Float('0.17999999999999999', prec=15)*y_1 + sympy.Float('0.029999999999999999', prec=15)) + M_0_0_3*(y_1**2*y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(6*c_0**4*(1 + y_2**2/c_0**2)**3) - y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))*(y_1**2 + M_0_2_0)/(6*c_0**4*(1 + y_2**2/c_0**2)**3)) + M_0_1_2*(y_1*(sympy.Float('-4.0', prec=15) + sympy.Float('16.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) - y_1*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(c_0**2*(1 + y_2**2/c_0**2)**2)) - (-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2))*(y_1**2 + M_0_2_0) - 2*(y_0*y_1 + M_1_1_0)*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) - sympy.Float('2.0', prec=15)*y_2*M_0_2_1/(c_0**2*(1 + y_2**2/c_0**2)**2)],
        [c_3*y_0**2*y_1 + c_3*M_2_1_0 - sympy.Float('0.089999999999999997', prec=15)*y_0**2*y_2 + 2*y_0*y_2*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2)) + sympy.Float('0.029999999999999999', prec=15)*y_0*y_2 + sympy.Float('2.0', prec=15)*y_0*y_2/(1 + y_2**2/c_0**2) + 2*y_0*(y_0*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) + y_2*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2))) - 2*y_0*(c_3*y_0*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_0*y_2 + sympy.Float('1.0', prec=15)*y_2/(1 + y_2**2/c_0**2)) - y_2*(-sympy.Float('0.059999999999999998', prec=15)*y_0**2 + sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('2.0', prec=15)*y_0/(1 + y_2**2/c_0**2) + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2)) + sympy.Float('1.0', prec=15)*y_2/(1 + y_2**2/c_0**2) + M_0_0_2*(-y_2*(y_0*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(c_0**2*(1 + y_2**2/c_0**2)**2) + (sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2)) + 2*y_0*y_2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(c_0**2*(1 + y_2**2/c_0**2)**2) + y_2*(sympy.Float('-6.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) - (sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))*(y_0*y_2 + M_1_0_1)/(c_0**2*(1 + y_2**2/c_0**2)**2)) + M_1_0_2*(y_2*(sympy.Float('-6.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(c_0**2*(1 + y_2**2/c_0**2)**2) - y_2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(c_0**2*(1 + y_2**2/c_0**2)**2)) - sympy.Float('0.089999999999999997', prec=15)*M_2_0_1 + M_1_0_1*(-sympy.Float('0.059999999999999998', prec=15)*y_0 + sympy.Float('0.029999999999999999', prec=15) + 2*(sympy.Float('1.0', prec=15) - sympy.Float('2.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(1 + y_2**2/c_0**2) + sympy.Float('4.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)**2)) + M_2_0_0*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2) + M_0_0_3*(-y_2*(y_0*y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(3*c_0**4*(1 + y_2**2/c_0**2)**3) + y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(6*c_0**4*(1 + y_2**2/c_0**2)**3)) + (sympy.Float('-6.0', prec=15) + sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)) - sympy.Float('48.0', prec=15)*y_2**4/(c_0**4*(1 + y_2**2/c_0**2)**2))/(6*c_0**2*(1 + y_2**2/c_0**2)**2) + 2*y_0*y_2**2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(3*c_0**4*(1 + y_2**2/c_0**2)**3) - y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))*(y_0*y_2 + M_1_0_1)/(3*c_0**4*(1 + y_2**2/c_0**2)**3)) - 2*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2))*(y_0*y_2 + M_1_0_1) - (y_0**2 + M_2_0_0)*(c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_2)],
        [c_2*y_0**3 + c_2*M_3_0_0 - c_3*y_0**2*y_1 - sympy.Float('0.089999999999999997', prec=15)*y_0**2*y_1 + 2*y_0*y_1*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2)) + sympy.Float('0.029999999999999999', prec=15)*y_0*y_1 + sympy.Float('2.0', prec=15)*y_0*y_1/(1 + y_2**2/c_0**2) + 2*y_0*(y_0*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) + y_1*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2))) - 2*y_0*(c_2*y_0**2 - c_3*y_0*y_1 - sympy.Float('0.059999999999999998', prec=15)*y_0*y_1 + sympy.Float('1.0', prec=15)*y_1/(1 + y_2**2/c_0**2)) - y_1*(-sympy.Float('0.059999999999999998', prec=15)*y_0**2 + sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('2.0', prec=15)*y_0/(1 + y_2**2/c_0**2) + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2)) + sympy.Float('1.0', prec=15)*y_1/(1 + y_2**2/c_0**2) + M_0_0_2*(-y_1*(y_0*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(c_0**2*(1 + y_2**2/c_0**2)**2) + (sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2)) + 2*y_0*y_1*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(c_0**2*(1 + y_2**2/c_0**2)**2) + y_1*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) - (sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))*(y_0*y_1 + M_1_1_0)/(c_0**2*(1 + y_2**2/c_0**2)**2)) + M_2_1_0*(-c_3 + sympy.Float('-0.089999999999999997', prec=15)) + M_1_1_0*(-2*c_3*y_0 - 2*y_0*(-c_3 + sympy.Float('-0.059999999999999998', prec=15)) - sympy.Float('0.17999999999999999', prec=15)*y_0 + sympy.Float('0.029999999999999999', prec=15) + sympy.Float('2.0', prec=15)/(1 + y_2**2/c_0**2)) + M_2_0_0*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) + M_0_0_3*(-y_1*(y_0*y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(3*c_0**4*(1 + y_2**2/c_0**2)**3) + y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(6*c_0**4*(1 + y_2**2/c_0**2)**3)) + 2*y_0*y_1*y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(3*c_0**4*(1 + y_2**2/c_0**2)**3) + y_1*y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(6*c_0**4*(1 + y_2**2/c_0**2)**3) - y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))*(y_0*y_1 + M_1_1_0)/(3*c_0**4*(1 + y_2**2/c_0**2)**3)) - 2*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2))*(y_0*y_1 + M_1_1_0) - (y_0**2 + M_2_0_0)*(c_2*y_0 - c_3*y_1 - sympy.Float('0.029999999999999999', prec=15)*y_1) - sympy.Float('4.0', prec=15)*y_2*M_1_1_1/(c_0**2*(1 + y_2**2/c_0**2)**2) - sympy.Float('2.0', prec=15)*y_2*M_0_1_1/(c_0**2*(1 + y_2**2/c_0**2)**2) + M_0_1_2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2)],
        [-sympy.Float('0.089999999999999997', prec=15)*y_0**3 + 6*y_0**2*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2)) + sympy.Float('0.089999999999999997', prec=15)*y_0**2 + sympy.Float('3.0', prec=15)*y_0**2/(1 + y_2**2/c_0**2) - 3*y_0*(-sympy.Float('0.059999999999999998', prec=15)*y_0**2 + sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('2.0', prec=15)*y_0/(1 + y_2**2/c_0**2) + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2)) - sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('3.0', prec=15)*y_0/(1 + y_2**2/c_0**2) + M_0_0_2*(-3*y_0*(y_0*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(c_0**2*(1 + y_2**2/c_0**2)**2) + (sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2)) + 9*y_0**2*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) + 3*y_0*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) - 3*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))*(y_0**2 + M_2_0_0)/(2*c_0**2*(1 + y_2**2/c_0**2)**2) + (sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2)) + M_1_0_2*(3*y_0*(sympy.Float('-4.0', prec=15) + sympy.Float('16.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) - 3*y_0*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(c_0**2*(1 + y_2**2/c_0**2)**2) + 3*(sympy.Float('-2.0', prec=15) + sympy.Float('8.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2)) - sympy.Float('0.089999999999999997', prec=15)*M_3_0_0 + M_2_0_0*(-sympy.Float('0.090000000000000024', prec=15)*y_0 + sympy.Float('0.089999999999999997', prec=15) + sympy.Float('3.0', prec=15)/(1 + y_2**2/c_0**2)) + M_0_0_3*(-3*y_0*(y_0*y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(3*c_0**4*(1 + y_2**2/c_0**2)**3) + y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(6*c_0**4*(1 + y_2**2/c_0**2)**3)) + 3*y_0**2*y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**4*(1 + y_2**2/c_0**2)**3) + y_0*y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**4*(1 + y_2**2/c_0**2)**3) - y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))*(y_0**2 + M_2_0_0)/(2*c_0**4*(1 + y_2**2/c_0**2)**3) + y_2*(sympy.Float('24.0', prec=15) - sympy.Float('48.0', prec=15)*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(6*c_0**4*(1 + y_2**2/c_0**2)**3)) - 3*(-sympy.Float('0.029999999999999999', prec=15)*y_0 + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2))*(y_0**2 + M_2_0_0) + sympy.Float('1.0', prec=15)/(1 + y_2**2/c_0**2) - sympy.Float('6.0', prec=15)*y_2*M_2_0_1/(c_0**2*(1 + y_2**2/c_0**2)**2) - sympy.Float('6.0', prec=15)*y_2*M_1_0_1/(c_0**2*(1 + y_2**2/c_0**2)**2)]])




        expected_constants = sympy.Matrix([
        [c_0],
        [c_1],
        [c_2],
        [c_3]])



        expected = ODEProblem("MEA", expected_lhs, expected_mfk, expected_constants)
        mea = MomentExpansionApproximation(hes1_model, max_order=self.max_order)
        answer = mea.run()



        self.assertEqual(answer, expected)

    # #
    def test_run_p53(self):
        p53_model = Model(parameters=['c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6'],
                  species=['y_0', 'y_1', 'y_2'],
                  stoichiometry_matrix=[[1, -1, -1, 0, 0, 0],
                                        [0, 0, 0, 1, -1, 0],
                                        [0, 0, 0, 0, 1, -1]],
                  propensities=['c_0',
                                'c_1*y_0',
                                'c_2*y_2*y_0/(y_0+c_6)',
                                'c_3*y_0',
                                'c_4*y_1',
                                'c_5*y_2'])

        expected_lhs = [
        Moment(array([1, 0, 0]), symbol=y_0),
        Moment(array([0, 1, 0]), symbol=y_1),
        Moment(array([0, 0, 1]), symbol=y_2),
        Moment(array([0, 0, 2]), symbol=M_0_0_2),
        Moment(array([0, 1, 1]), symbol=M_0_1_1),
        Moment(array([0, 2, 0]), symbol=M_0_2_0),
        Moment(array([1, 0, 1]), symbol=M_1_0_1),
        Moment(array([1, 1, 0]), symbol=M_1_1_0),
        Moment(array([2, 0, 0]), symbol=M_2_0_0),
        Moment(array([0, 0, 3]), symbol=M_0_0_3),
        Moment(array([0, 1, 2]), symbol=M_0_1_2),
        Moment(array([0, 2, 1]), symbol=M_0_2_1),
        Moment(array([0, 3, 0]), symbol=M_0_3_0),
        Moment(array([1, 0, 2]), symbol=M_1_0_2),
        Moment(array([1, 1, 1]), symbol=M_1_1_1),
        Moment(array([1, 2, 0]), symbol=M_1_2_0),
        Moment(array([2, 0, 1]), symbol=M_2_0_1),
        Moment(array([2, 1, 0]), symbol=M_2_1_0),
        Moment(array([3, 0, 0]), symbol=M_3_0_0),
        ]

        expected_mfk = sympy.Matrix([
        [c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0) - c_2*y_2*M_3_0_0*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - c_2*y_2*M_2_0_0*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*M_2_0_1*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*M_1_0_1*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)],
        [c_3*y_0 - c_4*y_1],
        [c_4*y_1 - c_5*y_2],
        [2*c_4*y_1*y_2 + c_4*y_1 + 2*c_4*M_0_1_1 - 2*c_5*y_2**2 + c_5*y_2 - 2*c_5*M_0_0_2 - 2*y_2*(c_4*y_1 - c_5*y_2)],
        [c_3*y_0*y_2 + c_3*M_1_0_1 + c_4*y_1**2 - c_4*y_1*y_2 - c_4*y_1 + c_4*M_0_2_0 - c_5*y_1*y_2 - y_1*(c_4*y_1 - c_5*y_2) - y_2*(c_3*y_0 - c_4*y_1) + M_0_1_1*(-c_4 - c_5)],
        [2*c_3*y_0*y_1 + c_3*y_0 + 2*c_3*M_1_1_0 - 2*c_4*y_1**2 + c_4*y_1 - 2*c_4*M_0_2_0 - 2*y_1*(c_3*y_0 - c_4*y_1)],
        [c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) - c_2*y_0*M_0_0_2/(c_6 + y_0) - c_2*y_2*M_2_0_1*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*M_1_0_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_4*y_0*y_1 + c_4*M_1_1_0 - c_5*y_0*y_2 - y_0*(c_4*y_1 - c_5*y_2) - y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + M_1_0_1*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_5)],
        [c_0*y_1 - c_1*y_0*y_1 - c_2*y_0*y_1*y_2/(c_6 + y_0) - c_2*y_0*M_0_1_1/(c_6 + y_0) - c_2*y_2*M_2_1_0*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*M_1_1_1*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_3*y_0**2 + c_3*M_2_0_0 - c_4*y_0*y_1 - y_0*(c_3*y_0 - c_4*y_1) - y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + M_1_1_0*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_4)],
        [2*c_0*y_0 + c_0 - 2*c_1*y_0**2 + c_1*y_0 - 2*c_2*y_0**2*y_2/(c_6 + y_0) + c_2*y_0*y_2/(c_6 + y_0) - 2*y_0*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + M_2_0_1*(2*c_2*y_0*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2) + M_3_0_0*(2*c_2*y_0*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 2*c_2*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3) + M_1_0_1*(2*c_2*y_0*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)) + M_2_0_0*(-2*c_1 + 2*c_2*y_0*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2)],
        [3*c_4*y_1*y_2**2 + 3*c_4*y_1*y_2 + c_4*y_1 + 3*c_4*M_0_1_1 + 3*c_4*M_0_1_2 - 3*c_5*y_2**3 + 3*c_5*y_2**2 - c_5*y_2 - 3*c_5*M_0_0_3 + 6*y_2**2*(c_4*y_1 - c_5*y_2) - 3*y_2*(2*c_4*y_1*y_2 + c_4*y_1 - 2*c_5*y_2**2 + c_5*y_2) + M_0_0_2*(3*c_4*y_1 - 3*c_5*y_2 + 3*c_5) - 3*(y_2**2 + M_0_0_2)*(c_4*y_1 - c_5*y_2)],
        [c_3*y_0*y_2**2 + c_3*M_1_0_2 + 2*c_4*y_1**2*y_2 + c_4*y_1**2 - c_4*y_1*y_2**2 - 2*c_4*y_1*y_2 - c_4*y_1 + c_4*M_0_2_0 + 2*c_4*M_0_2_1 - 2*c_5*y_1*y_2**2 + c_5*y_1*y_2 + 2*y_1*y_2*(c_4*y_1 - c_5*y_2) - y_1*(2*c_4*y_1*y_2 + c_4*y_1 - 2*c_5*y_2**2 + c_5*y_2) + 2*y_2*(y_1*(c_4*y_1 - c_5*y_2) + y_2*(c_3*y_0 - c_4*y_1)) - 2*y_2*(c_3*y_0*y_2 + c_4*y_1**2 - c_4*y_1*y_2 - c_4*y_1 - c_5*y_1*y_2) + M_0_0_2*(c_3*y_0 - c_4*y_1) + M_0_1_1*(2*c_4*y_1 - 2*c_4*y_2 - 2*c_4 - 4*c_5*y_2 + c_5 - 2*y_2*(-c_4 - c_5)) + M_0_1_2*(-c_4 - 2*c_5) - (y_2**2 + M_0_0_2)*(c_3*y_0 - c_4*y_1) - 2*(c_4*y_1 - c_5*y_2)*(y_1*y_2 + M_0_1_1)],
        [2*c_3*y_0*y_1*y_2 + c_3*y_0*y_2 + 2*c_3*M_1_1_1 + c_3*M_1_0_1 + c_4*y_1**3 - 2*c_4*y_1**2*y_2 - 2*c_4*y_1**2 + c_4*y_1*y_2 + c_4*y_1 + c_4*M_0_3_0 - c_5*y_1**2*y_2 + 2*y_1*y_2*(c_3*y_0 - c_4*y_1) + 2*y_1*(y_1*(c_4*y_1 - c_5*y_2) + y_2*(c_3*y_0 - c_4*y_1)) - 2*y_1*(c_3*y_0*y_2 + c_4*y_1**2 - c_4*y_1*y_2 - c_4*y_1 - c_5*y_1*y_2) - y_2*(2*c_3*y_0*y_1 + c_3*y_0 - 2*c_4*y_1**2 + c_4*y_1) + M_0_1_1*(2*c_3*y_0 - 4*c_4*y_1 + c_4 - 2*c_5*y_1 - 2*y_1*(-c_4 - c_5)) + M_0_2_0*(c_4*y_1 - 2*c_4 - c_5*y_2) + M_0_2_1*(-2*c_4 - c_5) - (y_1**2 + M_0_2_0)*(c_4*y_1 - c_5*y_2) - 2*(c_3*y_0 - c_4*y_1)*(y_1*y_2 + M_0_1_1)],
        [3*c_3*y_0*y_1**2 + 3*c_3*y_0*y_1 + c_3*y_0 + 3*c_3*M_1_2_0 + 3*c_3*M_1_1_0 - 3*c_4*y_1**3 + 3*c_4*y_1**2 - c_4*y_1 - 3*c_4*M_0_3_0 + 6*y_1**2*(c_3*y_0 - c_4*y_1) - 3*y_1*(2*c_3*y_0*y_1 + c_3*y_0 - 2*c_4*y_1**2 + c_4*y_1) + M_0_2_0*(3*c_3*y_0 - 3*c_4*y_1 + 3*c_4) - 3*(y_1**2 + M_0_2_0)*(c_3*y_0 - c_4*y_1)],
        [c_0*y_2**2 - c_1*y_0*y_2**2 - c_2*y_0*y_2**3/(c_6 + y_0) - c_2*y_0*M_0_0_3/(c_6 + y_0) + 2*c_4*y_0*y_1*y_2 + c_4*y_0*y_1 + 2*c_4*M_1_1_1 + c_4*M_1_1_0 - 2*c_5*y_0*y_2**2 + c_5*y_0*y_2 + 2*y_0*y_2*(c_4*y_1 - c_5*y_2) - y_0*(2*c_4*y_1*y_2 + c_4*y_1 - 2*c_5*y_2**2 + c_5*y_2) + 2*y_2*(y_0*(c_4*y_1 - c_5*y_2) + y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))) - 2*y_2*(c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) + c_4*y_0*y_1 - c_5*y_0*y_2) + M_0_0_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + M_1_0_2*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_5) + M_2_0_1*(-c_2*y_2**2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*(y_2**2 + M_0_0_2)*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2) + M_3_0_0*(-c_2*y_2**3*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 + c_2*y_2*(y_2**2 + M_0_0_2)*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3) + M_1_0_1*(-2*c_1*y_2 - 5*c_2*y_2**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*(y_2**2 + M_0_0_2)*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + 2*c_4*y_1 - 4*c_5*y_2 + c_5 - 2*y_2*(-c_1 - 2*c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_5)) + M_2_0_0*(-c_2*y_2**3*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2*(y_2**2 + M_0_0_2)*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2) - (y_2**2 + M_0_0_2)*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) - 2*(c_4*y_1 - c_5*y_2)*(y_0*y_2 + M_1_0_1)],
        [c_0*y_1*y_2 - c_1*y_0*y_1*y_2 - c_2*y_0*y_1*y_2**2/(c_6 + y_0) - c_2*y_0*M_0_1_2/(c_6 + y_0) + c_3*y_0**2*y_2 + c_4*y_0*y_1**2 - c_4*y_0*y_1*y_2 - c_4*y_0*y_1 + c_4*M_1_2_0 - c_5*y_0*y_1*y_2 + y_0*(y_1*(c_4*y_1 - c_5*y_2) + y_2*(c_3*y_0 - c_4*y_1)) - y_0*(c_3*y_0*y_2 + c_4*y_1**2 - c_4*y_1*y_2 - c_4*y_1 - c_5*y_1*y_2) + y_1*(y_0*(c_4*y_1 - c_5*y_2) + y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))) - y_1*(c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) + c_4*y_0*y_1 - c_5*y_0*y_2) + y_2*(y_0*(c_3*y_0 - c_4*y_1) + y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))) - y_2*(c_0*y_1 - c_1*y_0*y_1 - c_2*y_0*y_1*y_2/(c_6 + y_0) + c_3*y_0**2 - c_4*y_0*y_1) + M_1_1_1*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_4 - c_5) + M_2_0_1*(-c_2*y_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*(y_0/(c_6 + y_0) - 1)*(y_1*y_2 + M_0_1_1)/(c_6 + y_0)**2 + c_3) + M_3_0_0*(-c_2*y_1*y_2**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 + c_2*y_2*(-y_0/(c_6 + y_0) + 1)*(y_1*y_2 + M_0_1_1)/(c_6 + y_0)**3) + M_0_1_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0) - c_4*y_0 - c_5*y_0 - y_0*(-c_4 - c_5)) + M_1_0_1*(-c_1*y_1 - 3*c_2*y_1*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)*(y_1*y_2 + M_0_1_1)/(c_6 + y_0) + c_3*y_0 - c_4*y_1 - c_5*y_1 - y_1*(-c_1 - 2*c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_5)) + M_1_1_0*(-c_1*y_2 - c_2*y_2**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_4*y_1 - c_4*y_2 - c_4 - c_5*y_2 - y_2*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_4)) + M_2_0_0*(-2*c_2*y_1*y_2**2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2*(y_0/(c_6 + y_0) - 1)*(y_1*y_2 + M_0_1_1)/(c_6 + y_0)**2 + c_3*y_2 - y_2*(-c_2*y_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_3)) - (c_3*y_0 - c_4*y_1)*(y_0*y_2 + M_1_0_1) - (c_4*y_1 - c_5*y_2)*(y_0*y_1 + M_1_1_0) - (y_1*y_2 + M_0_1_1)*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))],
        [c_0*y_1**2 - c_1*y_0*y_1**2 - c_2*y_0*y_1**2*y_2/(c_6 + y_0) - c_2*y_0*M_0_2_1/(c_6 + y_0) + 2*c_3*y_0**2*y_1 + c_3*y_0**2 + 2*c_3*M_2_1_0 - 2*c_4*y_0*y_1**2 + c_4*y_0*y_1 + 2*y_0*y_1*(c_3*y_0 - c_4*y_1) - y_0*(2*c_3*y_0*y_1 + c_3*y_0 - 2*c_4*y_1**2 + c_4*y_1) + 2*y_1*(y_0*(c_3*y_0 - c_4*y_1) + y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))) - 2*y_1*(c_0*y_1 - c_1*y_0*y_1 - c_2*y_0*y_1*y_2/(c_6 + y_0) + c_3*y_0**2 - c_4*y_0*y_1) + M_1_2_0*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_4) + M_2_0_1*(-c_2*y_1**2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*(y_1**2 + M_0_2_0)*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2) + M_3_0_0*(-c_2*y_1**2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 + c_2*y_2*(y_1**2 + M_0_2_0)*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3) + M_0_2_0*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + M_1_0_1*(-c_2*y_1**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*(y_1**2 + M_0_2_0)*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)) + M_1_1_0*(-2*c_1*y_1 - 2*c_2*y_1*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + 2*c_3*y_0 - 4*c_4*y_1 + c_4 - 2*y_1*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_4)) + M_2_0_0*(-3*c_2*y_1**2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2*(y_1**2 + M_0_2_0)*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + 2*c_3*y_1 + c_3 - 2*y_1*(-c_2*y_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_3)) - (y_1**2 + M_0_2_0)*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) - 2*(c_3*y_0 - c_4*y_1)*(y_0*y_1 + M_1_1_0)],
        [2*c_0*y_0*y_2 + c_0*y_2 - 2*c_1*y_0**2*y_2 + c_1*y_0*y_2 - 2*c_2*y_0**2*y_2**2/(c_6 + y_0) + c_2*y_0*y_2**2/(c_6 + y_0) + c_2*y_0*M_0_0_2/(c_6 + y_0) + c_4*y_0**2*y_1 + c_4*M_2_1_0 - c_5*y_0**2*y_2 + 2*y_0*y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + 2*y_0*(y_0*(c_4*y_1 - c_5*y_2) + y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))) - 2*y_0*(c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) + c_4*y_0*y_1 - c_5*y_0*y_2) - y_2*(2*c_0*y_0 + c_0 - 2*c_1*y_0**2 + c_1*y_0 - 2*c_2*y_0**2*y_2/(c_6 + y_0) + c_2*y_0*y_2/(c_6 + y_0)) + M_1_0_2*(2*c_2*y_0*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)) + M_2_0_1*(-2*c_1 - 4*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + 2*c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + 2*c_2*(y_0*y_2 + M_1_0_1)*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_5 - y_2*(-2*c_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2)) + M_3_0_0*(-2*c_2*y_0*y_2**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 2*c_2*y_2**2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 + 2*c_2*y_2*(y_0*y_2 + M_1_0_1)*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - y_2*(-2*c_2*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3)) + M_1_0_1*(2*c_0 - 4*c_1*y_0 + c_1 - 4*c_2*y_0*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 4*c_2*y_0*y_2*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + 2*c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + 2*c_2*(y_0*y_2 + M_1_0_1)*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_5*y_0 - 2*y_0*(-c_1 - 2*c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_5) - y_2*(-2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0))) + M_2_0_0*(-2*c_1*y_2 - 2*c_2*y_0*y_2**2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_2**2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2**2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + 2*c_2*y_2*(y_0*y_2 + M_1_0_1)*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_4*y_1 - c_5*y_2 - y_2*(-2*c_1 - 2*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2)) - (y_0**2 + M_2_0_0)*(c_4*y_1 - c_5*y_2) - 2*(y_0*y_2 + M_1_0_1)*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))],
        [2*c_0*y_0*y_1 + c_0*y_1 - 2*c_1*y_0**2*y_1 + c_1*y_0*y_1 - 2*c_2*y_0**2*y_1*y_2/(c_6 + y_0) + c_2*y_0*y_1*y_2/(c_6 + y_0) + c_2*y_0*M_0_1_1/(c_6 + y_0) + c_3*y_0**3 - c_4*y_0**2*y_1 + 2*y_0*y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + 2*y_0*(y_0*(c_3*y_0 - c_4*y_1) + y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))) - 2*y_0*(c_0*y_1 - c_1*y_0*y_1 - c_2*y_0*y_1*y_2/(c_6 + y_0) + c_3*y_0**2 - c_4*y_0*y_1) - y_1*(2*c_0*y_0 + c_0 - 2*c_1*y_0**2 + c_1*y_0 - 2*c_2*y_0**2*y_2/(c_6 + y_0) + c_2*y_0*y_2/(c_6 + y_0)) + M_1_1_1*(2*c_2*y_0*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)) + M_2_0_1*(-2*c_2*y_0*y_1*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_1*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_1*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + 2*c_2*(y_0*y_1 + M_1_1_0)*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - y_1*(-2*c_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2)) + M_2_1_0*(-2*c_1 + 2*c_2*y_0*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_4) + M_3_0_0*(-2*c_2*y_0*y_1*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 2*c_2*y_1*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_1*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 + 2*c_2*y_2*(y_0*y_1 + M_1_1_0)*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 + c_3 - y_1*(-2*c_2*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3)) + M_1_0_1*(-2*c_2*y_0*y_1*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_2*y_0*y_1*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*y_1*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + 2*c_2*(y_0*y_1 + M_1_1_0)*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - y_1*(-2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0))) + M_1_1_0*(2*c_0 - 4*c_1*y_0 + c_1 - 2*c_2*y_0*y_2*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_4*y_0 - 2*y_0*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_4)) + M_2_0_0*(-2*c_1*y_1 - 4*c_2*y_0*y_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_1*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + 2*c_2*y_2*(y_0*y_1 + M_1_1_0)*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + 3*c_3*y_0 - c_4*y_1 - 2*y_0*(-c_2*y_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_3) - y_1*(-2*c_1 - 2*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2)) - (y_0**2 + M_2_0_0)*(c_3*y_0 - c_4*y_1) - 2*(y_0*y_1 + M_1_1_0)*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))],
        [3*c_0*y_0**2 + 3*c_0*y_0 + c_0 - 3*c_1*y_0**3 + 3*c_1*y_0**2 - c_1*y_0 - 3*c_2*y_0**3*y_2/(c_6 + y_0) + 3*c_2*y_0**2*y_2/(c_6 + y_0) - c_2*y_0*y_2/(c_6 + y_0) + 6*y_0**2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) - 3*y_0*(2*c_0*y_0 + c_0 - 2*c_1*y_0**2 + c_1*y_0 - 2*c_2*y_0**2*y_2/(c_6 + y_0) + c_2*y_0*y_2/(c_6 + y_0)) + M_2_0_1*(-6*c_2*y_0**2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 3*c_2*y_0*(y_0**2/(c_6 + y_0)**2 - 3*y_0/(c_6 + y_0) + 3)/(c_6 + y_0) + 3*c_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + 3*c_2*(y_0**2 + M_2_0_0)*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 3*y_0*(-2*c_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2)) + M_3_0_0*(-3*c_1 - 6*c_2*y_0**2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 3*c_2*y_2*(-y_0**3/(c_6 + y_0)**3 + 3*y_0**2/(c_6 + y_0)**2 - 3*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + 3*c_2*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + 3*c_2*y_2*(y_0**2 + M_2_0_0)*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 3*y_0*(-2*c_2*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3)) + M_1_0_1*(-6*c_2*y_0**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 3*c_2*y_0**2*(-y_0/(c_6 + y_0) + 3)/(c_6 + y_0) + 3*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + 3*c_2*(y_0**2 + M_2_0_0)*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 3*y_0*(-2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0))) + M_2_0_0*(3*c_0 - 9*c_1*y_0 + 3*c_1 - 6*c_2*y_0**2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 3*c_2*y_0*y_2*(y_0**2/(c_6 + y_0)**2 - 3*y_0/(c_6 + y_0) + 3)/(c_6 + y_0) + 3*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + 3*c_2*y_2*(y_0**2 + M_2_0_0)*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 3*y_0*(-2*c_1 - 2*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2)) - 3*(y_0**2 + M_2_0_0)*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))],
        ])

        expected_constants = sympy.Matrix([
        [c_0],
        [c_1],
        [c_2],
        [c_3],
        [c_4],
        [c_5],
        [c_6]])


        expected = ODEProblem("MEA", expected_lhs, expected_mfk, expected_constants)

        mea = MomentExpansionApproximation(p53_model, max_order=self.max_order)
        answer = mea.run()

        self.assertEqual(answer, expected)


class TestMomentExpansionApproximationOnMaxOrderIsTwo(unittest.TestCase):
    max_order = 2
    def test_run_dimer(self):
        dimer_model = Model(parameters=['c_0', 'c_1', 'c_2'],
                           species=['y_0'],
                           stoichiometry_matrix=[[-2, 2]],
                           propensities=['c_0*y_0*(y_0-1)',
                                         'c_1*((1.0/2)*(c_2-y_0))'])

        expected_lhs = [
        Moment(array([1]), symbol=y_0),
        Moment(array([2]), symbol=M_2),
        ]

        expected_mfk = sympy.Matrix([
        [-2*c_0*y_0*(y_0 - 1) - 2*c_0*M_2 + 2*c_1*(0.5*c_2 - 0.5*y_0)],
        [-4*c_0*y_0**2*(y_0 - 1) + 4*c_0*y_0*(y_0 - 1) + 4*c_1*y_0*(0.5*c_2 - 0.5*y_0) + 4*c_1*(0.5*c_2 - 0.5*y_0) - 2*y_0*(-2*c_0*y_0*(y_0 - 1) + 2*c_1*(0.5*c_2 - 0.5*y_0)) + M_2*(4*c_0*y_0 - 4*c_0*(3*y_0 - 1) + 4*c_0 - 2.0*c_1)],
        ])

        expected_constants = sympy.Matrix([
        [c_0],
        [c_1],
        [c_2]])

        expected = ODEProblem("MEA", expected_lhs, expected_mfk, expected_constants)
        mea = MomentExpansionApproximation(dimer_model, max_order=2)
        answer = mea.run()

    def test_run_mm(self):
        mm_model = Model(parameters=['c_0', 'c_1', 'c_2'],
                               species=['y_0', 'y_1'],
                               propensities=['c_0*y_0*(120-301+y_0+y_1)',
                                             'c_1*(301-(y_0+y_1))',
                                             'c_2*(301-(y_0+y_1))'],
                               stoichiometry_matrix=[[-1, 1, 0],
                                                     [0, 0, 1]])

        expected_lhs = [
        Moment(array([1, 0]), symbol=y_0),
        Moment(array([0, 1]), symbol=y_1),
        Moment(array([0, 2]), symbol=M_0_2),
        Moment(array([1, 1]), symbol=M_1_1),
        Moment(array([2, 0]), symbol=M_2_0),
        ]

        expected_mfk = sympy.Matrix([
        [-c_0*y_0*(y_0 + y_1 - 181) - c_0*M_1_1 - c_0*M_2_0 + c_1*(-y_0 - y_1 + 301)],
        [c_2*(-y_0 - y_1 + 301)],
        [-2*c_2*M_0_2 - 2*c_2*M_1_1 + c_2*(-y_0 - y_1 + 301)],
        [-c_0*y_0*y_1*(y_0 + y_1 - 181) + c_1*y_1*(-y_0 - y_1 + 301) - c_2*M_2_0 - y_1*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301)) + M_0_2*(-c_0*y_0 - c_1) + M_1_1*(c_0*y_1 - c_0*(2*y_0 + 2*y_1 - 181) - c_1 - c_2)],
        [-2*c_0*y_0**2*(y_0 + y_1 - 181) + c_0*y_0*(y_0 + y_1 - 181) + 2*c_1*y_0*(-y_0 - y_1 + 301) + c_1*(-y_0 - y_1 + 301) - 2*y_0*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301)) + M_1_1*(-2*c_0*y_0 + c_0 - 2*c_1) + M_2_0*(2*c_0*y_0 - 2*c_0*(3*y_0 + y_1 - 181) + c_0 - 2*c_1)],
        ])

        expected_constants = sympy.Matrix([
        [c_0],
        [c_1],
        [c_2]])


        expected = ODEProblem("MEA", expected_lhs, expected_mfk, expected_constants)
        mea = MomentExpansionApproximation(mm_model, max_order=self.max_order)
        answer = mea.run()

        self.assertEqual(answer, expected)

    def test_run_hes1(self):
        hes1_model = Model(parameters=['c_0', 'c_1', 'c_2', 'c_3'],
                   species=['y_0', 'y_1', 'y_2'],
                   propensities=['0.03*y_0',
                                 '0.03*y_1',
                                 '0.03*y_2',
                                 'c_3*y_1',
                                 'c_2*y_0',
                                 '1.0/(1+(y_2/c_0)**2)'],
                   stoichiometry_matrix=[[-1, 0, 0, 0, 0, 1],
                                         [0, -1, 0, -1, 1, 0],
                                         [0, 0, -1, 1, 0, 0]])


        expected_lhs = [
        Moment(array([1, 0, 0]), symbol=y_0),
        Moment(array([0, 1, 0]), symbol=y_1),
        Moment(array([0, 0, 1]), symbol=y_2),
        Moment(array([0, 0, 2]), symbol=M_0_0_2),
        Moment(array([0, 1, 1]), symbol=M_0_1_1),
        Moment(array([0, 2, 0]), symbol=M_0_2_0),
        Moment(array([1, 0, 1]), symbol=M_1_0_1),
        Moment(array([1, 1, 0]), symbol=M_1_1_0),
        Moment(array([2, 0, 0]), symbol=M_2_0_0)
        ]

        expected_mfk = sympy.Matrix([
        [-0.03*y_0 + 1.0/(1 + y_2**2/c_0**2) + M_0_0_2*(-2.0 + 8.0*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2)],
        [c_2*y_0 - c_3*y_1 - 0.03*y_1],
        [c_3*y_1 - 0.03*y_2],
        [2*c_3*y_1*y_2 + c_3*y_1 + 2*c_3*M_0_1_1 - 0.06*y_2**2 - 2*y_2*(c_3*y_1 - 0.03*y_2) + 0.03*y_2 - 0.06*M_0_0_2],
        [c_2*y_0*y_2 + c_2*M_1_0_1 + c_3*y_1**2 - c_3*y_1*y_2 - c_3*y_1 + c_3*M_0_2_0 - 0.06*y_1*y_2 - y_1*(c_3*y_1 - 0.03*y_2) - y_2*(c_2*y_0 - c_3*y_1 - 0.03*y_1) + M_0_1_1*(-c_3 - 0.06)],
        [2*c_2*y_0*y_1 + c_2*y_0 + 2*c_2*M_1_1_0 - 2*c_3*y_1**2 + c_3*y_1 - 0.06*y_1**2 - 2*y_1*(c_2*y_0 - c_3*y_1 - 0.03*y_1) + 0.03*y_1 + M_0_2_0*(-2*c_3 - 0.06)],
        [c_3*y_0*y_1 + c_3*M_1_1_0 - 0.06*y_0*y_2 - y_0*(c_3*y_1 - 0.03*y_2) - y_2*(-0.03*y_0 + 1.0/(1 + y_2**2/c_0**2)) + 1.0*y_2/(1 + y_2**2/c_0**2) + M_0_0_2*(y_2*(-6.0 + 8.0*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2) - y_2*(-2.0 + 8.0*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2)) - 0.06*M_1_0_1],
        [c_2*y_0**2 + c_2*M_2_0_0 - c_3*y_0*y_1 - 0.06*y_0*y_1 - y_0*(c_2*y_0 - c_3*y_1 - 0.03*y_1) - y_1*(-0.03*y_0 + 1.0/(1 + y_2**2/c_0**2)) + 1.0*y_1/(1 + y_2**2/c_0**2) + M_1_1_0*(-c_3 - 0.06) - 2.0*y_2*M_0_1_1/(c_0**2*(1 + y_2**2/c_0**2)**2)],
        [-0.06*y_0**2 - 2*y_0*(-0.03*y_0 + 1.0/(1 + y_2**2/c_0**2)) + 0.03*y_0 + 2.0*y_0/(1 + y_2**2/c_0**2) - 0.06*M_2_0_0 + 1.0/(1 + y_2**2/c_0**2) - 4.0*y_2*M_1_0_1/(c_0**2*(1 + y_2**2/c_0**2)**2) + M_0_0_2*(-2.0 + 8.0*y_2**2/(c_0**2*(1 + y_2**2/c_0**2)))/(2*c_0**2*(1 + y_2**2/c_0**2)**2)]
        ])

        expected_constants = sympy.Matrix([
        [c_0],
        [c_1],
        [c_2],
        [c_3]])


        expected = ODEProblem("MEA", expected_lhs, expected_mfk, expected_constants)
        mea = MomentExpansionApproximation(hes1_model, max_order=self.max_order)
        answer = mea.run()

        self.assertEqual(answer, expected)


    def test_run_p53(self):
        p53_model = Model(parameters=['c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6'],
                  species=['y_0', 'y_1', 'y_2'],
                  stoichiometry_matrix=[[1, -1, -1, 0, 0, 0],
                                        [0, 0, 0, 1, -1, 0],
                                        [0, 0, 0, 0, 1, -1]],
                  propensities=['c_0',
                                'c_1*y_0',
                                'c_2*y_2*y_0/(y_0+c_6)',
                                'c_3*y_0',
                                'c_4*y_1',
                                'c_5*y_2'])

        expected_lhs = [
        Moment(array([1, 0, 0]), symbol="y_0"),
        Moment(array([0, 1, 0]), symbol="y_1"),
        Moment(array([0, 0, 1]), symbol="y_2"),
        Moment(array([0, 0, 2]), symbol="M_0_0_2"),
        Moment(array([0, 1, 1]), symbol="M_0_1_1"),
        Moment(array([0, 2, 0]), symbol="M_0_2_0"),
        Moment(array([1, 0, 1]), symbol="M_1_0_1"),
        Moment(array([1, 1, 0]), symbol="M_1_1_0"),
        Moment(array([2, 0, 0]), symbol="M_2_0_0")
        ]

        expected_mfk = sympy.Matrix([
        [c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0) - c_2*y_2*M_2_0_0*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*M_1_0_1*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)],
        [c_3*y_0 - c_4*y_1],
        [c_4*y_1 - c_5*y_2],
        [2*c_4*y_1*y_2 + c_4*y_1 + 2*c_4*M_0_1_1 - 2*c_5*y_2**2 + c_5*y_2 - 2*c_5*M_0_0_2 - 2*y_2*(c_4*y_1 - c_5*y_2)],
        [c_3*y_0*y_2 + c_3*M_1_0_1 + c_4*y_1**2 - c_4*y_1*y_2 - c_4*y_1 + c_4*M_0_2_0 - c_5*y_1*y_2 - y_1*(c_4*y_1 - c_5*y_2) - y_2*(c_3*y_0 - c_4*y_1) + M_0_1_1*(-c_4 - c_5)],
        [2*c_3*y_0*y_1 + c_3*y_0 + 2*c_3*M_1_1_0 - 2*c_4*y_1**2 + c_4*y_1 - 2*c_4*M_0_2_0 - 2*y_1*(c_3*y_0 - c_4*y_1)],
        [c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) - c_2*y_0*M_0_0_2/(c_6 + y_0) + c_4*y_0*y_1 + c_4*M_1_1_0 - c_5*y_0*y_2 - y_0*(c_4*y_1 - c_5*y_2) - y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + M_1_0_1*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_5)],
        [c_0*y_1 - c_1*y_0*y_1 - c_2*y_0*y_1*y_2/(c_6 + y_0) - c_2*y_0*M_0_1_1/(c_6 + y_0) + c_3*y_0**2 + c_3*M_2_0_0 - c_4*y_0*y_1 - y_0*(c_3*y_0 - c_4*y_1) - y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + M_1_1_0*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_4)],
        [2*c_0*y_0 + c_0 - 2*c_1*y_0**2 + c_1*y_0 - 2*c_2*y_0**2*y_2/(c_6 + y_0) + c_2*y_0*y_2/(c_6 + y_0) - 2*y_0*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + M_1_0_1*(2*c_2*y_0*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)) + M_2_0_0*(-2*c_1 + 2*c_2*y_0*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2)]
        ])


        expected_constants = sympy.MutableDenseMatrix([[c_0], [c_1], [c_2], [c_3], [c_4], [c_5], [c_6]])
        expected = ODEProblem("MEA", expected_lhs, expected_mfk, expected_constants)

        mea = MomentExpansionApproximation(p53_model, max_order=self.max_order)
        answer = mea.run()

        self.assertEqual(answer, expected)

class TestMomentExpansionApproximation(unittest.TestCase):


    def test_substitute_raw_with_central(self):
        n_counter = [
            Moment([0, 0, 0], symbol=sympy.Integer(1)),
            Moment([0, 0, 2], symbol=sympy.Symbol("yx2")),
            Moment([0, 1, 1], symbol=sympy.Symbol("yx3")),
            Moment([0, 2, 0], symbol=sympy.Symbol("yx4")),
            Moment([1, 0, 1], symbol=sympy.Symbol("yx5")),
            Moment([1, 1, 0], symbol=sympy.Symbol("yx6")),
            Moment([2, 0, 0], symbol=sympy.Symbol("yx7")),
            Moment([0, 0, 3], symbol=sympy.Symbol("yx8")),
            Moment([0, 1, 2], symbol=sympy.Symbol("yx9")),
            Moment([0, 2, 1], symbol=sympy.Symbol("yx10")),
            Moment([0, 3, 0], symbol=sympy.Symbol("yx11")),
            Moment([1, 0, 2], symbol=sympy.Symbol("yx12")),
            Moment([1, 1, 1], symbol=sympy.Symbol("yx13")),
            Moment([1, 2, 0], symbol=sympy.Symbol("yx14")),
            Moment([2, 0, 1], symbol=sympy.Symbol("yx15")),
            Moment([2, 1, 0], symbol=sympy.Symbol("yx16")),
            Moment([3, 0, 0], symbol=sympy.Symbol("yx17"))
            ]
        k_counter = [
            Moment([0, 0, 0], symbol=sympy.Integer(1)),
            Moment([1, 0, 0], symbol=sympy.Symbol("y_0")),
            Moment([0, 1, 0], symbol=sympy.Symbol("y_1")),
            Moment([0, 0, 1], symbol=sympy.Symbol("y_2")),
            Moment([0, 0, 2], symbol=sympy.Symbol("x_0_0_2")),
            Moment([0, 1, 1], symbol=sympy.Symbol("x_0_1_1")),
            Moment([0, 2, 0], symbol=sympy.Symbol("x_0_2_0")),
            Moment([1, 0, 1], symbol=sympy.Symbol("x_1_0_1")),
            Moment([1, 1, 0], symbol=sympy.Symbol("x_1_1_0")),
            Moment([2, 0, 0], symbol=sympy.Symbol("x_2_0_0")),
            Moment([0, 0, 3], symbol=sympy.Symbol("x_0_0_3")),
            Moment([0, 1, 2], symbol=sympy.Symbol("x_0_1_2")),
            Moment([0, 2, 1], symbol=sympy.Symbol("x_0_2_1")),
            Moment([0, 3, 0], symbol=sympy.Symbol("x_0_3_0")),
            Moment([1, 0, 2], symbol=sympy.Symbol("x_1_0_2")),
            Moment([1, 1, 1], symbol=sympy.Symbol("x_1_1_1")),
            Moment([1, 2, 0], symbol=sympy.Symbol("x_1_2_0")),
            Moment([2, 0, 1], symbol=sympy.Symbol("x_2_0_1")),
            Moment([2, 1, 0], symbol=sympy.Symbol("x_2_1_0")),
            Moment([3, 0, 0], symbol=sympy.Symbol("x_3_0_0"))
        ]

        central_from_raw_exprs = to_sympy_matrix(
                    [["x_0_0_2-y_2**2"],
                    ["x_0_1_1-y_1*y_2"],
                    ["x_0_2_0-y_1**2"],
                    ["x_1_0_1-y_0*y_2"],
                    ["x_1_1_0-y_0*y_1"],
                    ["x_2_0_0-y_0**2"],
                    ["-3*x_0_0_2*y_2+x_0_0_3+2*y_2**3"],
                    ["-x_0_0_2*y_1-2*x_0_1_1*y_2+x_0_1_2+2*y_1*y_2**2"],
                    ["-2*x_0_1_1*y_1-x_0_2_0*y_2+x_0_2_1+2*y_1**2*y_2"],
                    ["-3*x_0_2_0*y_1+x_0_3_0+2*y_1**3"],
                    ["-x_0_0_2*y_0-2*x_1_0_1*y_2+x_1_0_2+2*y_0*y_2**2"],
                    ["-x_0_1_1*y_0-x_1_0_1*y_1-x_1_1_0*y_2+x_1_1_1+2*y_0*y_1*y_2"],
                    ["-x_0_2_0*y_0-2*x_1_1_0*y_1+x_1_2_0+2*y_0*y_1**2"],
                    ["-2*x_1_0_1*y_0-x_2_0_0*y_2+x_2_0_1+2*y_0**2*y_2"],
                    ["-2*x_1_1_0*y_0-x_2_0_0*y_1+x_2_1_0+2*y_0**2*y_1"],
                    ["-3*x_2_0_0*y_0+x_3_0_0+2*y_0**3"]
         ])
        c_4 = sympy.Symbol('c_4')
        y_1 = sympy.Symbol('y_1')
        y_2 = sympy.Symbol('y_2')
        c_5 = sympy.Symbol('c_5')
        c_3 = sympy.Symbol('c_3')
        y_0 = sympy.Symbol('y_0')
        c_0 = sympy.Symbol('c_0')
        c_1 = sympy.Symbol('c_1')
        c_2 = sympy.Symbol('c_2')
        c_6 = sympy.Symbol('c_6')
        x_0_0_2 = sympy.Symbol('x_0_0_2')
        x_0_1_1 = sympy.Symbol('x_0_1_1')
        x_0_2_0 = sympy.Symbol('x_0_2_0')
        x_1_0_1 = sympy.Symbol('x_1_0_1')
        x_1_1_0 = sympy.Symbol('x_1_1_0')
        x_2_0_0 = sympy.Symbol('x_2_0_0')
        central_moments_exprs = sympy.MutableDenseMatrix([
            [2*c_4*y_1*y_2 + c_4*y_1 - 2*c_5*y_2**2 + c_5*y_2 - 2*y_2*(c_4*y_1 - c_5*y_2), -2*c_5, 2*c_4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [c_3*y_0*y_2 + c_4*y_1**2 - c_4*y_1*y_2 - c_4*y_1 - c_5*y_1*y_2 - y_1*(c_4*y_1 - c_5*y_2) - y_2*(c_3*y_0 - c_4*y_1), 0, -c_4 - c_5, c_4, c_3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2*c_3*y_0*y_1 + c_3*y_0 - 2*c_4*y_1**2 + c_4*y_1 - 2*y_1*(c_3*y_0 - c_4*y_1), 0, 0, -2*c_4, 0, 2*c_3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) + c_4*y_0*y_1 - c_5*y_0*y_2 - y_0*(c_4*y_1 - c_5*y_2) - y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)), -c_2*y_0/(c_6 + y_0), 0, 0, -c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_5, c_4, 0, 0, 0, 0, 0, -c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0), 0, 0, -c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2, 0, 0],
            [c_0*y_1 - c_1*y_0*y_1 - c_2*y_0*y_1*y_2/(c_6 + y_0) + c_3*y_0**2 - c_4*y_0*y_1 - y_0*(c_3*y_0 - c_4*y_1) - y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)), 0, -c_2*y_0/(c_6 + y_0), 0, 0, -c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_4, c_3, 0, 0, 0, 0, 0, -c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0), 0, 0, -c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2, 0],

            [2*c_0*y_0 + c_0 - 2*c_1*y_0**2 + c_1*y_0 - 2*c_2*y_0**2*y_2/(c_6 + y_0) + c_2*y_0*y_2/(c_6 + y_0) - 2*y_0*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)), 0, 0, 0, 2*c_2*y_0*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0), 0, -2*c_1 + 2*c_2*y_0*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2, 0, 0, 0, 0, 0, 0, 0, 2*c_2*y_0*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2, 0, 2*c_2*y_0*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 2*c_2*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3],
            [3*c_4*y_1*y_2**2 + 3*c_4*y_1*y_2 + c_4*y_1 - 3*c_5*y_2**3 + 3*c_5*y_2**2 - c_5*y_2 - 3*x_0_0_2*(c_4*y_1 - c_5*y_2) + 6*y_2**2*(c_4*y_1 - c_5*y_2) - 3*y_2*(2*c_4*y_1*y_2 + c_4*y_1 - 2*c_5*y_2**2 + c_5*y_2), 3*c_4*y_1 - 3*c_5*y_2 + 3*c_5, 3*c_4, 0, 0, 0, 0, -3*c_5, 3*c_4, 0, 0, 0, 0, 0, 0, 0, 0]
            , [c_3*y_0*y_2**2 + 2*c_4*y_1**2*y_2 + c_4*y_1**2 - c_4*y_1*y_2**2 - 2*c_4*y_1*y_2 - c_4*y_1 - 2*c_5*y_1*y_2**2 + c_5*y_1*y_2 - x_0_0_2*(c_3*y_0 - c_4*y_1) - 2*x_0_1_1*(c_4*y_1 - c_5*y_2) + 2*y_1*y_2*(c_4*y_1 - c_5*y_2) - y_1*(2*c_4*y_1*y_2 + c_4*y_1 - 2*c_5*y_2**2 + c_5*y_2) + 2*y_2*(y_1*(c_4*y_1 - c_5*y_2) + y_2*(c_3*y_0 - c_4*y_1)) - 2*y_2*(c_3*y_0*y_2 + c_4*y_1**2 - c_4*y_1*y_2 - c_4*y_1 - c_5*y_1*y_2), c_3*y_0 - c_4*y_1, 2*c_4*y_1 - 2*c_4*y_2 - 2*c_4 - 4*c_5*y_2 + c_5 - 2*y_2*(-c_4 - c_5), c_4, 0, 0, 0, 0, -c_4 - 2*c_5, 2*c_4, 0, c_3, 0, 0, 0, 0, 0],
            [2*c_3*y_0*y_1*y_2 + c_3*y_0*y_2 + c_4*y_1**3 - 2*c_4*y_1**2*y_2 - 2*c_4*y_1**2 + c_4*y_1*y_2 + c_4*y_1 - c_5*y_1**2*y_2 - 2*x_0_1_1*(c_3*y_0 - c_4*y_1) - x_0_2_0*(c_4*y_1 - c_5*y_2) + 2*y_1*y_2*(c_3*y_0 - c_4*y_1) + 2*y_1*(y_1*(c_4*y_1 - c_5*y_2) + y_2*(c_3*y_0 - c_4*y_1)) - 2*y_1*(c_3*y_0*y_2 + c_4*y_1**2 - c_4*y_1*y_2 - c_4*y_1 - c_5*y_1*y_2) - y_2*(2*c_3*y_0*y_1 + c_3*y_0 - 2*c_4*y_1**2 + c_4*y_1), 0, 2*c_3*y_0 - 4*c_4*y_1 + c_4 - 2*c_5*y_1 - 2*y_1*(-c_4 - c_5), c_4*y_1 - 2*c_4 - c_5*y_2, c_3, 0, 0, 0, 0, -2*c_4 - c_5, c_4, 0, 2*c_3, 0, 0, 0, 0],
            [3*c_3*y_0*y_1**2 + 3*c_3*y_0*y_1 + c_3*y_0 - 3*c_4*y_1**3 + 3*c_4*y_1**2 - c_4*y_1 - 3*x_0_2_0*(c_3*y_0 - c_4*y_1) + 6*y_1**2*(c_3*y_0 - c_4*y_1) - 3*y_1*(2*c_3*y_0*y_1 + c_3*y_0 - 2*c_4*y_1**2 + c_4*y_1), 0, 0, 3*c_3*y_0 - 3*c_4*y_1 + 3*c_4, 0, 3*c_3, 0, 0, 0, 0, -3*c_4, 0, 0, 3*c_3, 0, 0, 0]
            , [c_0*y_2**2 - c_1*y_0*y_2**2 - c_2*y_0*y_2**3/(c_6 + y_0) + 2*c_4*y_0*y_1*y_2 + c_4*y_0*y_1 - 2*c_5*y_0*y_2**2 + c_5*y_0*y_2 - x_0_0_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) - 2*x_1_0_1*(c_4*y_1 - c_5*y_2) + 2*y_0*y_2*(c_4*y_1 - c_5*y_2) - y_0*(2*c_4*y_1*y_2 + c_4*y_1 - 2*c_5*y_2**2 + c_5*y_2) + 2*y_2*(y_0*(c_4*y_1 - c_5*y_2) + y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))) - 2*y_2*(c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) + c_4*y_0*y_1 - c_5*y_0*y_2), c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0), 0, 0, -2*c_1*y_2 + c_2*x_0_0_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 5*c_2*y_2**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + 2*c_4*y_1 - 4*c_5*y_2 + c_5 - 2*y_2*(-c_1 - 2*c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_5), c_4, c_2*x_0_0_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*y_2**3*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2, -c_2*y_0/(c_6 + y_0), 0, 0, 0, -c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_5, 2*c_4, 0, c_2*x_0_0_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*y_2**2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2, 0, c_2*x_0_0_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - c_2*y_2**3*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3], [c_0*y_1*y_2 - c_1*y_0*y_1*y_2 - c_2*y_0*y_1*y_2**2/(c_6 + y_0) + c_3*y_0**2*y_2 + c_4*y_0*y_1**2 - c_4*y_0*y_1*y_2 - c_4*y_0*y_1 - c_5*y_0*y_1*y_2 - x_0_1_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) - x_1_0_1*(c_3*y_0 - c_4*y_1) - x_1_1_0*(c_4*y_1 - c_5*y_2) + y_0*(y_1*(c_4*y_1 - c_5*y_2) + y_2*(c_3*y_0 - c_4*y_1)) - y_0*(c_3*y_0*y_2 + c_4*y_1**2 - c_4*y_1*y_2 - c_4*y_1 - c_5*y_1*y_2) + y_1*(y_0*(c_4*y_1 - c_5*y_2) + y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))) - y_1*(c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) + c_4*y_0*y_1 - c_5*y_0*y_2) + y_2*(y_0*(c_3*y_0 - c_4*y_1) + y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))) - y_2*(c_0*y_1 - c_1*y_0*y_1 - c_2*y_0*y_1*y_2/(c_6 + y_0) + c_3*y_0**2 - c_4*y_0*y_1), 0, c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0) - c_4*y_0 - c_5*y_0 - y_0*(-c_4 - c_5), 0, -c_1*y_1 + c_2*x_0_1_1*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 3*c_2*y_1*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_3*y_0 - c_4*y_1 - c_5*y_1 - y_1*(-c_1 - 2*c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_5), -c_1*y_2 - c_2*y_2**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_4*y_1 - c_4*y_2 - c_4 - c_5*y_2 - y_2*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_4), c_2*x_0_1_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_1*y_2**2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_3*y_2 - y_2*(-c_2*y_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_3), 0, -c_2*y_0/(c_6 + y_0), 0, 0, 0, -c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_4 - c_5, c_4, c_2*x_0_1_1*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*y_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_3, 0, c_2*x_0_1_1*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - c_2*y_1*y_2**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3], [c_0*y_1**2 - c_1*y_0*y_1**2 - c_2*y_0*y_1**2*y_2/(c_6 + y_0) + 2*c_3*y_0**2*y_1 + c_3*y_0**2 - 2*c_4*y_0*y_1**2 + c_4*y_0*y_1 - x_0_2_0*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) - 2*x_1_1_0*(c_3*y_0 - c_4*y_1) + 2*y_0*y_1*(c_3*y_0 - c_4*y_1) - y_0*(2*c_3*y_0*y_1 + c_3*y_0 - 2*c_4*y_1**2 + c_4*y_1) + 2*y_1*(y_0*(c_3*y_0 - c_4*y_1) + y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))) - 2*y_1*(c_0*y_1 - c_1*y_0*y_1 - c_2*y_0*y_1*y_2/(c_6 + y_0) + c_3*y_0**2 - c_4*y_0*y_1), 0, 0, c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0), c_2*x_0_2_0*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_2*y_1**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0), -2*c_1*y_1 - 2*c_2*y_1*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + 2*c_3*y_0 - 4*c_4*y_1 + c_4 - 2*y_1*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_4), c_2*x_0_2_0*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 3*c_2*y_1**2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + 2*c_3*y_1 + c_3 - 2*y_1*(-c_2*y_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_3), 0, 0, -c_2*y_0/(c_6 + y_0), 0, 0, 0, -c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_4, c_2*x_0_2_0*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*y_1**2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2, 2*c_3, c_2*x_0_2_0*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - c_2*y_1**2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3], [2*c_0*y_0*y_2 + c_0*y_2 - 2*c_1*y_0**2*y_2 + c_1*y_0*y_2 - 2*c_2*y_0**2*y_2**2/(c_6 + y_0) + c_2*y_0*y_2**2/(c_6 + y_0) + c_4*y_0**2*y_1 - c_5*y_0**2*y_2 - 2*x_1_0_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) - x_2_0_0*(c_4*y_1 - c_5*y_2) + 2*y_0*y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + 2*y_0*(y_0*(c_4*y_1 - c_5*y_2) + y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))) - 2*y_0*(c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) + c_4*y_0*y_1 - c_5*y_0*y_2) - y_2*(2*c_0*y_0 + c_0 - 2*c_1*y_0**2 + c_1*y_0 - 2*c_2*y_0**2*y_2/(c_6 + y_0) + c_2*y_0*y_2/(c_6 + y_0)), c_2*y_0/(c_6 + y_0), 0, 0, 2*c_0 - 4*c_1*y_0 + c_1 + 2*c_2*x_1_0_1*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 4*c_2*y_0*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 4*c_2*y_0*y_2*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + 2*c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_5*y_0 - 2*y_0*(-c_1 - 2*c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_5) - y_2*(-2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)), 0, -2*c_1*y_2 + 2*c_2*x_1_0_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_0*y_2**2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_2**2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2**2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_4*y_1 - c_5*y_2 - y_2*(-2*c_1 - 2*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2), 0, 0, 0, 0, 2*c_2*y_0*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0), 0, 0, -2*c_1 + 2*c_2*x_1_0_1*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 4*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + 2*c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_5 - y_2*(-2*c_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2), c_4, 2*c_2*x_1_0_1*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 2*c_2*y_0*y_2**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 2*c_2*y_2**2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - y_2*(-2*c_2*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3)], [2*c_0*y_0*y_1 + c_0*y_1 - 2*c_1*y_0**2*y_1 + c_1*y_0*y_1 - 2*c_2*y_0**2*y_1*y_2/(c_6 + y_0) + c_2*y_0*y_1*y_2/(c_6 + y_0) + c_3*y_0**3 - c_4*y_0**2*y_1 - 2*x_1_1_0*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) - x_2_0_0*(c_3*y_0 - c_4*y_1) + 2*y_0*y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + 2*y_0*(y_0*(c_3*y_0 - c_4*y_1) + y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))) - 2*y_0*(c_0*y_1 - c_1*y_0*y_1 - c_2*y_0*y_1*y_2/(c_6 + y_0) + c_3*y_0**2 - c_4*y_0*y_1) - y_1*(2*c_0*y_0 + c_0 - 2*c_1*y_0**2 + c_1*y_0 - 2*c_2*y_0**2*y_2/(c_6 + y_0) + c_2*y_0*y_2/(c_6 + y_0)), 0, c_2*y_0/(c_6 + y_0), 0, 2*c_2*x_1_1_0*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_2*y_0*y_1*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_2*y_0*y_1*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*y_1*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - y_1*(-2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)), 2*c_0 - 4*c_1*y_0 + c_1 - 2*c_2*y_0*y_2*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_4*y_0 - 2*y_0*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_4), -2*c_1*y_1 + 2*c_2*x_1_1_0*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 4*c_2*y_0*y_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_1*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + 3*c_3*y_0 - c_4*y_1 - 2*y_0*(-c_2*y_1*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_3) - y_1*(-2*c_1 - 2*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2), 0, 0, 0, 0, 0, 2*c_2*y_0*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0), 0, 2*c_2*x_1_1_0*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_0*y_1*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_1*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_1*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - y_1*(-2*c_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2), -2*c_1 + 2*c_2*y_0*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_4, 2*c_2*x_1_1_0*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 2*c_2*y_0*y_1*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 2*c_2*y_1*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_1*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 + c_3 - y_1*(-2*c_2*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3)], [3*c_0*y_0**2 + 3*c_0*y_0 + c_0 - 3*c_1*y_0**3 + 3*c_1*y_0**2 - c_1*y_0 - 3*c_2*y_0**3*y_2/(c_6 + y_0) + 3*c_2*y_0**2*y_2/(c_6 + y_0) - c_2*y_0*y_2/(c_6 + y_0) - 3*x_2_0_0*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + 6*y_0**2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) - 3*y_0*(2*c_0*y_0 + c_0 - 2*c_1*y_0**2 + c_1*y_0 - 2*c_2*y_0**2*y_2/(c_6 + y_0) + c_2*y_0*y_2/(c_6 + y_0)), 0, 0, 0, 3*c_2*x_2_0_0*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 6*c_2*y_0**2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 3*c_2*y_0**2*(-y_0/(c_6 + y_0) + 3)/(c_6 + y_0) + 3*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) - c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 3*y_0*(-2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)), 0, 3*c_0 - 9*c_1*y_0 + 3*c_1 + 3*c_2*x_2_0_0*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 6*c_2*y_0**2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 3*c_2*y_0*y_2*(y_0**2/(c_6 + y_0)**2 - 3*y_0/(c_6 + y_0) + 3)/(c_6 + y_0) + 3*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 3*y_0*(-2*c_1 - 2*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2), 0, 0, 0, 0, 0, 0, 0, 3*c_2*x_2_0_0*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 6*c_2*y_0**2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 3*c_2*y_0*(y_0**2/(c_6 + y_0)**2 - 3*y_0/(c_6 + y_0) + 3)/(c_6 + y_0) + 3*c_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 3*y_0*(-2*c_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2), 0, -3*c_1 + 3*c_2*x_2_0_0*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 6*c_2*y_0**2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 3*c_2*y_2*(-y_0**3/(c_6 + y_0)**3 + 3*y_0**2/(c_6 + y_0)**2 - 3*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + 3*c_2*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 3*y_0*(-2*c_2*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3)]
        ])

        yx2 = sympy.Symbol('yx2')
        yx3 = sympy.Symbol('yx3')
        yx4 = sympy.Symbol('yx4')
        yx5 = sympy.Symbol('yx5')
        yx6 = sympy.Symbol('yx6')
        yx7 = sympy.Symbol('yx7')
        expected = sympy.MutableDenseMatrix([[c_4*y_1 + c_5*y_2, -2*c_5, 2*c_4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [-c_4*y_1, 0, -c_4 - c_5, c_4, c_3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [c_3*y_0 + c_4*y_1, 0, 0, -2*c_4, 0, 2*c_3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, -c_2*y_0/(c_6 + y_0), 0, 0, -c_1 - c_2*c_6*y_2/(c_6 + y_0)**2 - c_5, c_4, 0, 0, 0, 0, 0, -c_2*c_6/(c_6 + y_0)**2, 0, 0, c_2*c_6*y_2/(c_6 + y_0)**3, 0, 0], [0, 0, -c_2*y_0/(c_6 + y_0), 0, 0, -c_1 - c_2*c_6*y_2/(c_6 + y_0)**2 - c_4, c_3, 0, 0, 0, 0, 0, -c_2*c_6/(c_6 + y_0)**2, 0, 0, c_2*c_6*y_2/(c_6 + y_0)**3, 0], [(c_0*c_6 + c_0*y_0 + c_1*c_6*y_0 + c_1*y_0**2 + c_2*y_0*y_2)/(c_6 + y_0), 0, 0, 0, c_2*(2*c_6*y_0 + c_6 - 2*y_0*(2*c_6 + y_0))/(c_6 + y_0)**2, 0, -(2*c_1*(c_6 + y_0)**3 + c_2*c_6*y_2*(2*y_0 + 1) + 2*c_2*y_2*(y_0**2 - 2*y_0*(c_6 + y_0) + (c_6 + y_0)**2))/(c_6 + y_0)**3, 0, 0, 0, 0, 0, 0, 0, c_2*(-c_6*(2*y_0 + 1) - 2*y_0**2 + 4*y_0*(c_6 + y_0) - 2*(c_6 + y_0)**2)/(c_6 + y_0)**3, 0, c_2*y_2*(c_6*(2*y_0 + 1) + 2*y_0**2 - 4*y_0*(c_6 + y_0) + 2*(c_6 + y_0)**2)/(c_6 + y_0)**4], [-3*c_4*y_1*yx2 + c_4*y_1 + 3*c_5*y_2*yx2 - c_5*y_2, 3*c_4*y_1 - 3*c_5*y_2 + 3*c_5, 3*c_4, 0, 0, 0, 0, -3*c_5, 3*c_4, 0, 0, 0, 0, 0, 0, 0, 0], [-c_3*y_0*yx2 + c_4*y_1*yx2 - 2*c_4*y_1*yx3 - c_4*y_1 + 2*c_5*y_2*yx3, c_3*y_0 - c_4*y_1, 2*c_4*y_1 - 2*c_4 - 2*c_5*y_2 + c_5, c_4, 0, 0, 0, 0, -c_4 - 2*c_5, 2*c_4, 0, c_3, 0, 0, 0, 0, 0], [-2*c_3*y_0*yx3 + 2*c_4*y_1*yx3 - c_4*y_1*yx4 + c_4*y_1 + c_5*y_2*yx4, 0, 2*c_3*y_0 - 2*c_4*y_1 + c_4, c_4*y_1 - 2*c_4 - c_5*y_2, c_3, 0, 0, 0, 0, -2*c_4 - c_5, c_4, 0, 2*c_3, 0, 0, 0, 0], [-3*c_3*y_0*yx4 + c_3*y_0 + 3*c_4*y_1*yx4 - c_4*y_1, 0, 0, 3*c_3*y_0 - 3*c_4*y_1 + 3*c_4, 0, 3*c_3, 0, 0, 0, 0, -3*c_4, 0, 0, 3*c_3, 0, 0, 0], [(-c_0*c_6*yx2 - c_0*y_0*yx2 + c_1*c_6*y_0*yx2 + c_1*y_0**2*yx2 + c_2*y_0*y_2*yx2 - 2*c_4*c_6*y_1*yx5 - 2*c_4*y_0*y_1*yx5 + 2*c_5*c_6*y_2*yx5 + 2*c_5*y_0*y_2*yx5)/(c_6 + y_0), (-c_2*y_0*y_2 + (c_0 - c_1*y_0)*(c_6 + y_0))/(c_6 + y_0), 0, 0, c_2*c_6*yx2/(c_6 + y_0)**2 + 2*c_4*y_1 - 2*c_5*y_2 + c_5, c_4, -c_2*c_6*y_2*yx2/(c_6 + y_0)**3, -c_2*y_0/(c_6 + y_0), 0, 0, 0, -c_1 - c_2*c_6*y_2/(c_6 + y_0)**2 - 2*c_5, 2*c_4, 0, -c_2*c_6*yx2/(c_6 + y_0)**3, 0, c_2*c_6*y_2*yx2/(c_6 + y_0)**4], [(-c_0*c_6*yx3 - c_0*y_0*yx3 + c_1*c_6*y_0*yx3 + c_1*y_0**2*yx3 + c_2*y_0*y_2*yx3 - c_3*c_6*y_0*yx5 - c_3*y_0**2*yx5 + c_4*c_6*y_1*yx5 - c_4*c_6*y_1*yx6 + c_4*y_0*y_1*yx5 - c_4*y_0*y_1*yx6 + c_5*c_6*y_2*yx6 + c_5*y_0*y_2*yx6)/(c_6 + y_0), 0, (c_0*c_6 + c_0*y_0 - c_1*c_6*y_0 - c_1*y_0**2 - c_2*y_0*y_2)/(c_6 + y_0), 0, c_2*c_6*yx3/(c_6 + y_0)**2 + c_3*y_0 - c_4*y_1, c_4*y_1 - c_4 - c_5*y_2, -c_2*c_6*y_2*yx3/(c_6 + y_0)**3, 0, -c_2*y_0/(c_6 + y_0), 0, 0, 0, -c_1 - c_2*c_6*y_2/(c_6 + y_0)**2 - c_4 - c_5, c_4, -c_2*c_6*yx3/(c_6 + y_0)**3 + c_3, 0, c_2*c_6*y_2*yx3/(c_6 + y_0)**4], [(-c_0*c_6*yx4 - c_0*y_0*yx4 + c_1*c_6*y_0*yx4 + c_1*y_0**2*yx4 + c_2*y_0*y_2*yx4 - 2*c_3*c_6*y_0*yx6 - 2*c_3*y_0**2*yx6 + 2*c_4*c_6*y_1*yx6 + 2*c_4*y_0*y_1*yx6)/(c_6 + y_0), 0, 0, (-c_2*y_0*y_2 + (c_0 - c_1*y_0)*(c_6 + y_0))/(c_6 + y_0), c_2*c_6*yx4/(c_6 + y_0)**2, 2*c_3*y_0 - 2*c_4*y_1 + c_4, -c_2*c_6*y_2*yx4/(c_6 + y_0)**3 + c_3, 0, 0, -c_2*y_0/(c_6 + y_0), 0, 0, 0, -c_1 - c_2*c_6*y_2/(c_6 + y_0)**2 - 2*c_4, -c_2*c_6*yx4/(c_6 + y_0)**3, 2*c_3, c_2*c_6*y_2*yx4/(c_6 + y_0)**4], [(-2*c_0*c_6*yx5 - 2*c_0*y_0*yx5 + 2*c_1*c_6*y_0*yx5 + 2*c_1*y_0**2*yx5 + 2*c_2*y_0*y_2*yx5 - c_4*c_6*y_1*yx7 - c_4*y_0*y_1*yx7 + c_5*c_6*y_2*yx7 + c_5*y_0*y_2*yx7)/(c_6 + y_0), c_2*y_0/(c_6 + y_0), 0, 0, 2*c_0 - 2*c_1*y_0 + c_1 - 2*c_2*c_6*y_0*y_2/(c_6 + y_0)**2 + c_2*c_6*y_2/(c_6 + y_0)**2 + 2*c_2*c_6*yx5/(c_6 + y_0)**2 - 2*c_2*y_0**2*y_2/(c_6 + y_0)**2, 0, -2*c_2*c_6*y_2*yx5/(c_6 + y_0)**3 + c_4*y_1 - c_5*y_2, 0, 0, 0, 0, c_2*(2*c_6*y_0 + c_6 - 2*y_0*(2*c_6 + y_0))/(c_6 + y_0)**2, 0, 0, -2*c_1 + 2*c_2*c_6*y_0*y_2/(c_6 + y_0)**3 - c_2*c_6*y_2/(c_6 + y_0)**3 - 2*c_2*c_6*yx5/(c_6 + y_0)**3 + 2*c_2*y_0**2*y_2/(c_6 + y_0)**3 - 2*c_2*y_2/(c_6 + y_0) - c_5, c_4, 2*c_2*c_6*y_2*yx5/(c_6**4 + 4*c_6**3*y_0 + 6*c_6**2*y_0**2 + 4*c_6*y_0**3 + y_0**4)], [(-2*c_0*c_6*yx6 - 2*c_0*y_0*yx6 + 2*c_1*c_6*y_0*yx6 + 2*c_1*y_0**2*yx6 + 2*c_2*y_0*y_2*yx6 - c_3*c_6*y_0*yx7 - c_3*y_0**2*yx7 + c_4*c_6*y_1*yx7 + c_4*y_0*y_1*yx7)/(c_6 + y_0), 0, c_2*y_0/(c_6 + y_0), 0, 2*c_2*c_6*yx6/(c_6**2 + 2*c_6*y_0 + y_0**2), 2*c_0 - 2*c_1*y_0 + c_1 - 2*c_2*c_6*y_0*y_2/(c_6 + y_0)**2 + c_2*c_6*y_2/(c_6 + y_0)**2 - 2*c_2*y_0**2*y_2/(c_6 + y_0)**2, (-2*c_2*c_6*y_2*yx6 + c_3*c_6**3*y_0 + 3*c_3*c_6**2*y_0**2 + 3*c_3*c_6*y_0**3 + c_3*y_0**4 - c_4*c_6**3*y_1 - 3*c_4*c_6**2*y_0*y_1 - 3*c_4*c_6*y_0**2*y_1 - c_4*y_0**3*y_1)/(c_6**3 + 3*c_6**2*y_0 + 3*c_6*y_0**2 + y_0**3), 0, 0, 0, 0, 0, c_2*(2*c_6*y_0 + c_6 - 2*y_0*(2*c_6 + y_0))/(c_6 + y_0)**2, 0, -2*c_2*c_6*yx6/(c_6**3 + 3*c_6**2*y_0 + 3*c_6*y_0**2 + y_0**3), -(c_2*c_6*y_2*(2*y_0 + 1) + 2*c_2*y_2*(y_0**2 - 2*y_0*(c_6 + y_0) + (c_6 + y_0)**2) + (2*c_1 + c_4)*(c_6 + y_0)**3)/(c_6 + y_0)**3, 2*c_2*c_6*y_2*yx6/(c_6 + y_0)**4 + c_3], [(-3*c_0*c_6*yx7 + c_0*c_6 - 3*c_0*y_0*yx7 + c_0*y_0 + 3*c_1*c_6*y_0*yx7 - c_1*c_6*y_0 + 3*c_1*y_0**2*yx7 - c_1*y_0**2 + 3*c_2*y_0*y_2*yx7 - c_2*y_0*y_2)/(c_6 + y_0), 0, 0, 0, c_2*(3*c_6*y_0 + 3*c_6*yx7 - c_6 + 3*y_0**2)/(c_6**2 + 2*c_6*y_0 + y_0**2), 0, 3*c_0 - 3*c_1*y_0 + 3*c_1 - 3*c_2*c_6*y_0*y_2/(c_6 + y_0)**3 - 3*c_2*c_6*y_2*yx7/(c_6 + y_0)**3 + c_2*c_6*y_2/(c_6 + y_0)**3 - 3*c_2*y_0**2*y_2/(c_6 + y_0)**3 - 3*c_2*y_0*y_2/(c_6 + y_0) + 3*c_2*y_2/(c_6 + y_0), 0, 0, 0, 0, 0, 0, 0, -c_2*(3*c_6**2*y_0 - 3*c_6**2 + 6*c_6*y_0**2 - 3*c_6*y_0 + 3*c_6*yx7 - c_6 + 3*y_0**3)/(c_6**3 + 3*c_6**2*y_0 + 3*c_6*y_0**2 + y_0**3), 0, -3*c_1 + 3*c_2*c_6*y_0*y_2/(c_6 + y_0)**4 + 3*c_2*c_6*y_2*yx7/(c_6 + y_0)**4 - c_2*c_6*y_2/(c_6 + y_0)**4 + 3*c_2*y_0**2*y_2/(c_6 + y_0)**4 + 3*c_2*y_0*y_2/(c_6 + y_0)**2 - 3*c_2*y_2/(c_6 + y_0) - 3*c_2*y_2/(c_6 + y_0)**2]])

        mea = MomentExpansionApproximation(None, 3)
        answer = mea._substitute_raw_with_central(central_moments_exprs, central_from_raw_exprs, n_counter, k_counter)

        self.assertTrue(sympy_expressions_equal(answer, expected))
