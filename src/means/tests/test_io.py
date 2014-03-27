import os
import unittest
from assimulo.solvers.sundials import CVodeError
from sympy import Symbol, MutableDenseMatrix, Float
from means import TrajectoryCollection, SolverException
from means.core import ODEProblem, Moment, VarianceTerm
from means.inference import Inference
from means.inference.results import InferenceResult, NormalConvergenceStatus
from means.io.serialise import dump, load
from means.examples.sample_models import MODEL_P53, MODEL_MICHAELIS_MENTEN, MODEL_LOTKA_VOLTERRA, \
                                         MODEL_HES1, MODEL_DIMERISATION
from means.simulation import Trajectory, TrajectoryWithSensitivityData, SensitivityTerm, Simulation
import numpy as np
from StringIO import StringIO
from tempfile import mkstemp
import cPickle as pickle

def _sample_problem():
    lhs_terms = [Moment(np.array([1, 0, 0]), symbol='y_0'),
                 Moment(np.array([0, 1, 0]), symbol='y_1'),
                 Moment(np.array([0, 0, 1]), symbol='y_2'),
                 Moment(np.array([0, 0, 2]), symbol='yx1'),
                 Moment(np.array([0, 1, 1]), symbol='yx2'),
                 Moment(np.array([0, 2, 0]), symbol='yx3'),
                 Moment(np.array([1, 0, 1]), symbol='yx4'),
                 Moment(np.array([1, 1, 0]), symbol='yx5'),
                 Moment(np.array([2, 0, 0]), symbol='yx6')]
    constants = ['c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6']

    c_0 = Symbol('c_0')
    c_1 = Symbol('c_1')
    y_0 = Symbol('y_0')
    c_2 = Symbol('c_2')
    y_2 = Symbol('y_2')
    c_6 = Symbol('c_6')
    yx4 = Symbol('yx4')
    yx6 = Symbol('yx6')
    c_3 = Symbol('c_3')
    c_4 = Symbol('c_4')
    y_1 = Symbol('y_1')
    c_5 = Symbol('c_5')
    yx2 = Symbol('yx2')
    yx1 = Symbol('yx1')
    yx3 = Symbol('yx3')
    yx5 = Symbol('yx5')
    rhs = MutableDenseMatrix([[c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0) + yx4*(c_2*y_0/(c_6 + y_0)**2 - c_2/(c_6 + y_0)) + yx6*(-c_2*y_0*y_2/(c_6 + y_0)**3 + c_2*y_2/(c_6 + y_0)**2)], [c_3*y_0 - c_4*y_1], [c_4*y_1 - c_5*y_2], [2*c_4*y_1*y_2 + c_4*y_1 + 2*c_4*yx2 - 2*c_5*y_2**2 + c_5*y_2 - 2*c_5*yx1 - 2*y_2*(c_4*y_1 - c_5*y_2)], [c_3*y_0*y_2 + c_3*yx4 + c_4*y_1**2 - c_4*y_1*y_2 - c_4*y_1 + c_4*yx3 - c_5*y_1*y_2 - y_1*(c_4*y_1 - c_5*y_2) - y_2*(c_3*y_0 - c_4*y_1) + yx2*(-c_4 - c_5)], [2*c_3*y_0*y_1 + c_3*y_0 + 2*c_3*yx5 - 2*c_4*y_1**2 + c_4*y_1 - 2*c_4*yx3 - 2*y_1*(c_3*y_0 - c_4*y_1)], [c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) - c_2*y_0*yx1/(c_6 + y_0) + c_4*y_0*y_1 + c_4*yx5 - c_5*y_0*y_2 - y_0*(c_4*y_1 - c_5*y_2) - y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + yx4*(-c_1 + 2*c_2*y_0*y_2/(c_6 + y_0)**2 - 2*c_2*y_2/(c_6 + y_0) - c_5 - y_2*(c_2*y_0/(c_6 + y_0)**2 - c_2/(c_6 + y_0))) + yx6*(-c_2*y_0*y_2**2/(c_6 + y_0)**3 + c_2*y_2**2/(c_6 + y_0)**2 - y_2*(-c_2*y_0*y_2/(c_6 + y_0)**3 + c_2*y_2/(c_6 + y_0)**2))], [c_0*y_1 - c_1*y_0*y_1 - c_2*y_0*y_1*y_2/(c_6 + y_0) - c_2*y_0*yx2/(c_6 + y_0) + c_3*y_0**2 - c_4*y_0*y_1 - y_0*(c_3*y_0 - c_4*y_1) - y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + yx4*(c_2*y_0*y_1/(c_6 + y_0)**2 - c_2*y_1/(c_6 + y_0) - y_1*(c_2*y_0/(c_6 + y_0)**2 - c_2/(c_6 + y_0))) + yx5*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0) - c_4) + yx6*(-c_2*y_0*y_1*y_2/(c_6 + y_0)**3 + c_2*y_1*y_2/(c_6 + y_0)**2 + c_3 - y_1*(-c_2*y_0*y_2/(c_6 + y_0)**3 + c_2*y_2/(c_6 + y_0)**2))], [2*c_0*y_0 + c_0 - 2*c_1*y_0**2 + c_1*y_0 - 2*c_2*y_0**2*y_2/(c_6 + y_0) + c_2*y_0*y_2/(c_6 + y_0) - 2*y_0*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + yx4*(2*c_2*y_0**2/(c_6 + y_0)**2 - 4*c_2*y_0/(c_6 + y_0) - c_2*y_0/(c_6 + y_0)**2 + c_2/(c_6 + y_0) - 2*y_0*(c_2*y_0/(c_6 + y_0)**2 - c_2/(c_6 + y_0))) + yx6*(-2*c_1 - 2*c_2*y_0**2*y_2/(c_6 + y_0)**3 + 4*c_2*y_0*y_2/(c_6 + y_0)**2 + c_2*y_0*y_2/(c_6 + y_0)**3 - 2*c_2*y_2/(c_6 + y_0) - c_2*y_2/(c_6 + y_0)**2 - 2*y_0*(-c_2*y_0*y_2/(c_6 + y_0)**3 + c_2*y_2/(c_6 + y_0)**2))]])

    problem = ODEProblem(method='MEA', left_hand_side_descriptors=lhs_terms, right_hand_side=rhs, parameters=constants)
    return problem

def _sample_inference():

    r = Inference(problem=_sample_problem(),
                      starting_parameters=[1, 2, 3, 4, 5, 6, 7],
                      starting_conditions=[1,2,3],
                      variable_parameters=['c_0', 'c_1'],
                      observed_trajectories=[Trajectory([1,2], [2,3], Moment([1, 0, 0], 'x'))],
                      distance_function_type='gamma',
                      maxh=0.01) # Some simulation kwargs
    return r

class TestSerialisation(unittest.TestCase):



    def _roundtrip(self, object_):
        # Check dump/load works
        self.assertEquals(load(dump(object_)), object_)

    def test_model_serialisation_works(self):
        """
        Given a model object, the serialisation routine should be able to dump that model and recover it when
        the dumped data is loaded
        :return:
        """

        self._roundtrip(MODEL_P53)
        self._roundtrip(MODEL_MICHAELIS_MENTEN)
        self._roundtrip(MODEL_LOTKA_VOLTERRA)
        self._roundtrip(MODEL_HES1)
        self._roundtrip(MODEL_DIMERISATION)

    def test_odeproblem_serialisation_works(self):
        problem = _sample_problem()
        self._roundtrip(problem)
        # Now make sure to access problem.right_hand_side_as_function as this sometimes breaks pickle
        f = problem.right_hand_side_as_function
        # Do roundtrip again
        self._roundtrip(problem)

    def test_ode_problem_lna_serialisation_works(self):

        c_0 = Symbol('c_0')
        c_1 = Symbol('c_1')
        y_0 = Symbol('y_0')
        c_2 = Symbol('c_2')
        y_2 = Symbol('y_2')
        c_6 = Symbol('c_6')
        c_3 = Symbol('c_3')
        c_4 = Symbol('c_4')
        y_1 = Symbol('y_1')
        c_5 = Symbol('c_5')
        V_00 = Symbol('V_00')
        V_02 = Symbol('V_02')
        V_20 = Symbol('V_20')
        V_01 = Symbol('V_01')
        V_21 = Symbol('V_21')
        V_22 = Symbol('V_22')
        V_10 = Symbol('V_10')
        V_12 = Symbol('V_12')
        V_11 = Symbol('V_11')
        right_hand_side = MutableDenseMatrix([[c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)], [c_3*y_0 - c_4*y_1], [c_4*y_1 - c_5*y_2], [2*V_00*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_02*c_2*y_0/(c_6 + y_0) - V_20*c_2*y_0/(c_6 + y_0) + c_0**Float('1.0', prec=15) + (c_1*y_0)**Float('1.0', prec=15) + (c_2*y_0*y_2/(c_6 + y_0))**Float('1.0', prec=15)], [V_00*c_3 - V_01*c_4 + V_01*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_21*c_2*y_0/(c_6 + y_0)], [V_01*c_4 - V_02*c_5 + V_02*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_22*c_2*y_0/(c_6 + y_0)], [V_00*c_3 - V_10*c_4 + V_10*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_12*c_2*y_0/(c_6 + y_0)], [V_01*c_3 + V_10*c_3 - 2*V_11*c_4 + (c_3*y_0)**Float('1.0', prec=15) + (c_4*y_1)**Float('1.0', prec=15)], [V_02*c_3 + V_11*c_4 - V_12*c_4 - V_12*c_5 - (c_4*y_1)**Float('1.0', prec=15)], [V_10*c_4 - V_20*c_5 + V_20*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_22*c_2*y_0/(c_6 + y_0)], [V_11*c_4 + V_20*c_3 - V_21*c_4 - V_21*c_5 - (c_4*y_1)**Float('1.0', prec=15)], [V_12*c_4 + V_21*c_4 - 2*V_22*c_5 + (c_4*y_1)**Float('1.0', prec=15) + (c_5*y_2)**Float('1.0', prec=15)]])

        ode_lhs_terms = [Moment(np.array([1, 0, 0]), symbol=y_0),
                         Moment(np.array([0, 1, 0]), symbol=y_1),
                         Moment(np.array([0, 0, 1]), symbol=y_2),
                         VarianceTerm((0, 0), V_00),
                         VarianceTerm((0, 1), V_01),
                         VarianceTerm((0, 2), V_02),
                         VarianceTerm((1, 0), V_10),
                         VarianceTerm((1, 1), V_11),
                         VarianceTerm((1, 2), V_12),
                         VarianceTerm((2, 0), V_20),
                         VarianceTerm((2, 1), V_21),
                         VarianceTerm((2, 2), V_22)]

        constants = ['c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6']

        problem = ODEProblem('LNA', ode_lhs_terms, right_hand_side, constants)
        self._roundtrip(problem)

        # Now make sure to access problem.right_hand_side_as_function as this sometimes breaks pickle
        f = problem.right_hand_side_as_function
        # Do roundtrip again
        self._roundtrip(problem)

    def test_trajectory_serialisation(self):
        t = Trajectory([1, 2, 3], [3, 2, 1], Moment([1,2,3], 'x'))
        self._roundtrip(t)

    def test_trajectory_collection_serialisation(self):

        t1 = Trajectory([1, 2, 3], [3, 2, 1], Moment([1,2,3], 'x'))
        t2 = Trajectory([1, 2, 3], [3, 2, 1], Moment([1,0,0], 'y'))
        tc = TrajectoryCollection([t1, t2])
        self._roundtrip(tc)

    def test_trajectory_with_sensitivities_serialisation(self):
        term = Moment([1, 0, 0], 'x')
        x = Trajectory([1, 2, 3], [3, 2, 1], SensitivityTerm(term, 'x'))
        y = Trajectory([1, 2, 3], [7, 8, 9], SensitivityTerm(term, 'y'))
        t = TrajectoryWithSensitivityData([1, 2, 3], [-1, -2, -3], term, sensitivity_data=[x, y])
        self._roundtrip(t)

    def test_inference_result_serialisation(self):
        r = InferenceResult(inference=_sample_inference(),
                            optimal_parameters=[15, 16, -17, 18, 19, 20],
                            optimal_initial_conditions=[-15, -16, -18],
                            distance_at_minimum=15.8,
                            convergence_status=NormalConvergenceStatus(None, 100, 180),
                            solutions=[[([1, 2, 3, 4, 5, 6], [3, 2, 1]), ([3, 2, 1, 0, -1, -2], [1, 2, 3])]],
                            distance_landscape=[([1, 2, 3, 4, 5, 6], [3, 2, 1], 15.4), ([3, 2, 1, 0, -1, -2], [1, 2, 3], 10),
                                                 ([3, 2, 1, 0, -1, -2], [1, 2, 3], 3.4)]
                            )
        self._roundtrip(r)

    def test_inference_serialisation(self):

        r = _sample_inference()
        self._roundtrip(r)


class TestSerialisationStringIO(TestSerialisation):
    def _roundtrip(self, object_):
        s = StringIO()
        try:
            object_.to_file(s)

            # Reset the file pointer so we can read from it
            s.seek(0)

            new_object = object_.__class__.from_file(s)
            self.assertEqual(object_, new_object)
        finally:
            s.close()


    def test_deserialisation_of_wrong_class_fails(self):
        """
        Given a file where a model object has been serialised, assert that
        deserialisation of that file into a trajectory object fails with ValueError
        :return:
        """

        s = StringIO()
        try:
            MODEL_P53.to_file(s)
            self.assertRaises(ValueError, Trajectory.from_file, s)
        finally:
            s.close()



class TestSerialisationFileIO(TestSerialisation):
    def _roundtrip(self, object_):
        __, tmp_file = mkstemp()
        try:
            object_.to_file(tmp_file)
            new_object = object_.__class__.from_file(tmp_file)
            self.assertEqual(object_, new_object)
        finally:
            os.unlink(tmp_file)


    def test_deserialisation_of_wrong_class_fails(self):
        """
        Given a file where a model object has been serialised, assert that
        deserialisation of that file into a trajectory object fails with ValueError
        :return:
        """

        __, tmp_file = mkstemp()
        try:
            MODEL_P53.to_file(tmp_file)
            self.assertRaises(ValueError, Trajectory.from_file, tmp_file)
        finally:
            os.unlink(tmp_file)

class TestPickleSerialisation(TestSerialisation):
    """
    We need make sure our classes are serialisable using pickle as well, as this is python default and much more
    compressed of a file format, than the human readable yaml format
    """

    def _roundtrip(self, object_):
        pickle_dump = pickle.dumps(object_, pickle.HIGHEST_PROTOCOL)
        new_object = pickle.loads(pickle_dump)

        self.assertEqual(object_, new_object)

    def test_solver_exception(self):

        e = SolverException(None, CVodeError('Test', 14.9))
        self._roundtrip(e)


