import os
import unittest
from numpy.testing import assert_array_almost_equal
import scipy.io.matlab
import means
import means.examples
import numpy as np
from means.simulation import SolverException

MODELS = {'p53': means.examples.MODEL_P53}

class TestTrajectoriesMatch(unittest.TestCase):


    def _read_data_from_matlab(self, matfile):
        """
        Returns the tajectories from matlab file provided as `matfile` argument
        :param matfile: a file.mat where the trajectory data is stored
        :return:
        """
        TRAJECTORIES_VARIABLE_NAME = 'trajectories'
        TIMEPOINTS_VARIABLE_NAME = 'timepoints'
        N_MOMENTS_VARIABLE_NAME = 'nMoments'
        PARAMETERS_VARIABLE_NAME = 'parameters'
        INITIAL_CONDITIONS_VARIABLE_NAME = 'init_val'
        MODEL_VARIABLE_NAME = 'model_name'
        CLOSURE_TYPE_VARIABLE_NAME = 'closure'
        CLOSURE_MULTIVARIATE_VARIABLE_NAME = 'multivariate'

        data = scipy.io.matlab.loadmat(matfile)
        return {'trajectories': data[TRAJECTORIES_VARIABLE_NAME],
                'n_moments': data[N_MOMENTS_VARIABLE_NAME],
                # Reshape the `initial_conditions`, `parameters`  and `timepoints` to be one-dimensional
                'parameters': data[PARAMETERS_VARIABLE_NAME].reshape(-1),
                'initial_conditions': data[INITIAL_CONDITIONS_VARIABLE_NAME].reshape(-1),
                'timepoints': data[TIMEPOINTS_VARIABLE_NAME].reshape(-1),
                # Scipy reads everything as arrays, even things that shouldn't be, thus [0]'s below
                'model_name': data[MODEL_VARIABLE_NAME][0],
                'closure': data[CLOSURE_TYPE_VARIABLE_NAME][0],
                'closure_is_multivariate': data[CLOSURE_MULTIVARIATE_VARIABLE_NAME][0]}

    def _compare_trajectories(self, our_trajectories, matlab_trajectories, only_the_first_n=None):

        # Check that we have similar number of trajectories
        self.assertEquals(len(our_trajectories), len(matlab_trajectories))

        for i, trajectory in enumerate(our_trajectories):
            if only_the_first_n is not None and i >= only_the_first_n:
                break

            matlab_trajectory = matlab_trajectories[i, :]
            assert_array_almost_equal(trajectory.values, matlab_trajectory, decimal=4)


    def _perform_test(self, matlab_filename):

        #-- Parse the data from MATLAB -------------------------
        data = self._read_data_from_matlab(matlab_filename)
        timepoints = data['timepoints']
        matlab_trajectories = data['trajectories']

        max_order = data['n_moments']  # We use one more moment than MATLAB for the same thing
        parameters = data['parameters']
        initial_conditions = data['initial_conditions']
        model_name = data['model_name']
        closure = data['closure']
        multivariate = data['closure_is_multivariate']

        #-- Do the test ---------------------------------------

        model = MODELS[model_name]
        problem = means.approximation.MomentExpansionApproximation(model,
                                                                   max_order=max_order,
                                                                   closer=closure,
                                                                   multivariate=multivariate).run()

        # The test script sets maxh equivalent to 0.01 in matlab, so let's do it here as well
        simulation = means.simulation.Simulation(problem, solver='ode15s', maxh=0.01)
        results = simulation.simulate_system(parameters, initial_conditions, timepoints)

        self._compare_trajectories(results, matlab_trajectories, problem.number_of_species)

    def test_p53_3_moments_lognormal_multivariate(self):
        self._perform_test(os.path.join(os.path.dirname(__file__), 'p53_3_moments_lognormal_multivariate.mat'))

class TestODE15SFailsWhereMatlabDoes(unittest.TestCase):

    def test_lognormal_2_mom_fails_early(self):

        problem = means.approximation.MomentExpansionApproximation(means.examples.MODEL_P53, 2, closer='log-normal')
        problem = problem.run()

        s = means.simulation.Simulation(problem, solver='ode15s', maxh=0.1)

        try:
            trajectories = s.simulate_system([90, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01], [70, 30, 60],
                                             np.arange(0, 40, 0.1))
        except SolverException as e:
            base_exception = e.base_exception
            # Check that the exception occured at timepoint similar to the timepoint in MATLAB
            self.assertAlmostEqual(base_exception.t, 17.35795, places=1)
        else:
            self.fail('ode15s was able to reach output without throwing and exception')


