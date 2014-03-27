"""
Simulate
--------

This part of the package provides utilities for simulate the
dynamic of an :class:`~means.core.problems.ODEProblem`.

A wide range of numerical solver are available:

>>> from means import Simulation
>>> print Simulation.supported_solvers()

In order to simulate a system, it is necessary to provide values for
the initial conditions and parameters (constants):

>>> from means import mea_approximation
>>> from means.examples.sample_models import MODEL_P53
>>> from means import Simulation
>>> import numpy as np
>>>
>>> ode_problem = mea_approximation(MODEL_P53,max_order=2)
>>> # We provide initial conditions, constants and time range
>>> RATES = [90, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01]
>>> INITIAL_CONDITIONS = [70, 30, 60]
>>> TMAX = 40
>>> TIME_RANGE = np.arange(0, TMAX, .1)
>>> #This is where we simulate the system to obtain trajectories
>>> simulator = Simulation(ode_problem)
>>> trajectories = simulator.simulate_system(RATES, INITIAL_CONDITIONS, TIME_RANGE)

A :class:`~means.simulation.trajectory.TrajectoryCollection` object (see :mod:`~means.simulation.trajectory`) is created.
See the documentation of :class:`Simulation` for additional information.

-------------
"""

import numpy as np
from means.io.serialise import SerialisableObject
from means.simulation.solvers import available_solvers
from means.simulation.trajectory import Trajectory, TrajectoryWithSensitivityData, TrajectoryCollection
from means.core import Moment, VarianceTerm

def _validate_problem(problem):

    problem.validate()

    if problem.method == "MEA":
        moments = filter(lambda x: isinstance(x, Moment), problem.left_hand_side_descriptors)
        if problem.left_hand_side.rows != len(moments):
            raise ValueError("There are {0} equations and {1} moments. "
                             "For MEA problems, the same number is expected.".format(problem.left_hand_side.rows,
                                                                                     len(moments)))
    elif problem.method == 'LNA':
        # FIXME: do some validation for LNA here
        pass



class Simulation(SerialisableObject):
    """
    Class that allows to perform simulations of the trajectories for a particular problem.
    Implements all ODE solvers supported by Assimulo_ package.

    .. _Assimulo: http://www.jmodelica.org/assimulo_home/

    """
    __problem = None
    _solver_options = None
    _solver = None

    yaml_tag = '!simulation'

    def __init__(self, problem, solver='ode15s', **solver_options):
        """

        :param problem: Problem to simulate
        :type problem: ODEProblem
        :param compute_sensitivities: Whether the model should test parameter sensitivity or not
        :param solver: the solver to use. Currently, the solvers that available are:

                       `'cvode'`
                            sundials CVode solver, as implemented in :class:`assimulo.solvers.sundials.CVode`
                       `'dopri5'`
                            Dopri5 solver, see :class:`assimulo.solvers.runge_kutta.Dopri5`
                       `'euler'`
                            Euler solver, see :class:`assimulo.solvers.euler.ExplicitEuler`
                       `'ode15s`:
                            sundials CVODE solver, with default parameters set to mimick the MATLAB's
                            See :class:`~means.simulation.solvers.ODE15sMixin` for the list of these parameters.
                       `'lsodar'`
                            LSODAR solver, see :class:`assimulo.solvers.odepack.LSODAR`
                       `'radau5'`
                            Radau5 solver, see :class:`assimulo.solvers.radau5.Radau5ODE`
                       `'rodas'`
                            Rosenbrock method of order (3)4 with step-size control,
                            see :class:`assimulo.solvers.rosenbrock.RodasODE`
                       `'rungekutta34'`
                            Adaptive Runda-Kutta of order four,
                            see :class:`assimulo.solvers.runge_kutta.RungeKutta34`
                       `'rungekutta4'`
                            Runge-Kutta method of order 4,
                            see :class:`assimulo.solvers.runge_kutta.RungeKutta4`

                       The list of these solvers is always accessible at runtime
                       from :meth:`Simulation.supported_solvers()` method.

                       .. _`ode15s`: http://www.mathworks.ch/ch/help/matlab/ref/ode15s.html

        :type solver: basestring
        :param solver_options: options to set in the solver. Consult `Assimulo documentation`_ for available options
                               for information on specific options available.

        .. _`Assimulo documentation`: http://www.jmodelica.org/assimulo_home/
        """
        self.__problem = problem
        _validate_problem(problem)


        self._solver = solver.lower()
        self._solver_options = solver_options

    def _append_zeros(self, initial_conditions, number_of_equations):
        """If not all intial conditions specified, append zeros to them
           TODO: is this really the best way to do this?
        """

        if len(initial_conditions) < number_of_equations:
            initial_conditions = np.concatenate((initial_conditions,
                                                 [0.0] * (self.problem.number_of_equations - len(initial_conditions))))
        return initial_conditions

    @classmethod
    def _supported_solvers_dict(cls):
        return available_solvers(with_sensitivity_support=False)

    @classmethod
    def supported_solvers(cls):
        """
        List the supported solvers for the simulations.

        >>> Simulation.supported_solvers()
        ['cvode', 'dopri5', 'euler', 'lsodar', 'radau5', 'rodas', 'rungekutta34', 'rungekutta4', 'ode15s']

        :return: the names of the solvers supported for simulations
        """
        return sorted(cls._supported_solvers_dict().keys())

    @property
    def _solver_class(self):
        supported_solvers = self._supported_solvers_dict()
        try:
            solver_class = supported_solvers[self._solver]
        except KeyError:
            raise Exception('Solver {0!r} not available. '
                            'Available solvers: {1!r}'.format(self._solver, self.supported_solvers()))

        return solver_class


    def _initialise_solver(self, initial_conditions, parameters, timepoints):

        solver = self._solver_class(self.problem, parameters, initial_conditions, starting_time=timepoints[0],
                                    **self._solver_options)
        return solver

    def simulate_system(self, parameters, initial_conditions, timepoints):
        """
        Simulates the system for each of the timepoints, starting at initial_constants and initial_values values

        :param parameters: list of the initial values for the constants in the model.
                                  Must be in the same order as in the model
        :param initial_conditions: List of the initial values for the equations in the problem. Must be in the same order as
                               these equations occur.
                               If not all values specified, the remaining ones will be assumed to be 0.
        :param timepoints: A list of time points to simulate the system for
        :return: a list of :class:`~means.simulation.Trajectory` objects,
                 one for each of the equations in the problem
        :rtype: list[:class:`~means.simulation.Trajectory`]
        """

        initial_conditions = self._append_zeros(initial_conditions, self.problem.number_of_equations)
        solver = self._initialise_solver(initial_conditions, parameters, timepoints)
        trajectories = solver.simulate(timepoints)

        return TrajectoryCollection(trajectories)

    @property
    def problem(self):
        return self.__problem

    @property
    def solver(self):
        return self._solver

    @property
    def solver_options(self):
        return self._solver_options

    @classmethod
    def to_yaml(cls, dumper, data):

        mapping = [('problem', data.problem),
                   ('solver', data._solver)]
        mapping.extend(data._solver_options.items())

        return dumper.represent_mapping(cls.yaml_tag, mapping)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.problem == other.problem and self.solver == other.solver \
            and self.solver_options == other.solver_options

class SimulationWithSensitivities(Simulation):
    """
    A similar class to it's baseclass :class:`~means.simulation.simulate.Simulation`.
    Performs simulations of the trajectories for each of the ODEs in given problem and performs sensitivity simulations
    for the problem's parameters.

    """

    def __init__(self, problem, solver='ode15s', **solver_options):
        """

        :param problem: Problem to simulate
        :type problem: ODEProblem
        :param compute_sensitivities: Whether the model should test parameter sensitivity or not
        :param solver: the solver to use. Currently, the solvers that available are:

                       `'cvode'`
                            sundials CVode solver, as implemented in :class:`assimulo.solvers.sundials.CVode`
                       `'ode15s`:
                            sundials CVODE solver, with default parameters set to mimick the MATLAB's
                            See :class:`~means.simulation.solvers.ODE15sMixin` for the list of these parameters.

                       .. _`ode15s`: http://www.mathworks.ch/ch/help/matlab/ref/ode15s.html

                       The list of these solvers is always accessible at runtime
                       from :meth:`SimulationWithSensitivities.supported_solvers()` method.

        :type solver: basestring
        :param solver_options: options to set in the solver. Consult `Assimulo documentation`_ for available options
                               for information on specific options available.

        .. _`Assimulo documentation`: http://www.jmodelica.org/assimulo_home/
        """
        super(SimulationWithSensitivities, self).__init__(problem, solver, **solver_options)

    @classmethod
    def _supported_solvers_dict(cls):
        return available_solvers(with_sensitivity_support=True)

    @classmethod
    def supported_solvers(cls):
        """
        List the supported solvers for the simulations.

        >>> SimulationWithSensitivities.supported_solvers()
        ['cvode', 'ode15s']

        :return: the names of the solvers supported for simulations
        """
        return super(SimulationWithSensitivities, cls).supported_solvers()


    def simulate_system(self, parameters, initial_conditions, timepoints):
        """
        Simulates the system for each of the timepoints, starting at initial_constants and initial_values values

        :param parameters: list of the initial values for the constants in the model.
                                  Must be in the same order as in the model
        :param initial_conditions: List of the initial values for the equations in the problem. Must be in the same order as
                               these equations occur.
                               If not all values specified, the remaining ones will be assumed to be 0.
        :param timepoints: A list of time points to simulate the system for
        :return: a list of :class:`~means.simulation.TrajectoryWithSensitivityData` objects,
                 one for each of the equations in the problem
        :rtype: list[:class:`~means.simulation.TrajectoryWithSensitivityData`]
        """
        return super(SimulationWithSensitivities, self).simulate_system(parameters, initial_conditions, timepoints)
