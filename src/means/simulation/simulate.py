"""
Simulates data for given model, moments, parameters, initial conditions
and method (moment expansion or LNA)
"""
from collections import namedtuple

from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import CVode
from assimulo.solvers.sundials import CVodeError
import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix


# These are the default values in solver.c but they seem very low
from means.approximation.ode_problem import Moment

RTOL = 1e-4
ATOL = 1e-4
NP_FLOATING_POINT_PRECISION = np.double

class Trajectory(object):
    """
    A single simulated or observed trajectory for an ODE term.
    """
    _timepoints = None
    _values = None
    _description = None

    def __init__(self, timepoints, values, description):
        """

        :param timepoints: timepoints the trajectory was simulated for
        :type timepoints: :class:`numpy.ndarray`
        :param values: values of the curve at each of the timepoints
        :type values: :class:`numpy.ndarray`
        :param description: description of the trajectory
        :type description: :class:`~means.approximation.ode_problem.ODETermBase`
        """
        self._timepoints = np.array(timepoints)
        self._values = np.array(values)
        self._description = description

    @property
    def timepoints(self):
        """
        The timepoints trajectory was simulated for.

        :rtype: :class:`numpy.ndarray`
        """
        return self._timepoints

    @property
    def values(self):
        """
        The values for each of the timepoints in :attr:`~Trajectory.timepoints`.

        :rtype: :class:`numpy.ndarray`
        """
        return self._values

    @property
    def description(self):
        """
        Description of this trajectory. The same description as the description for particular ODE term.

        :rtype: :class:`~means.approximation.ode_problem.ODETermBase`
        """
        return self._description

    def plot(self, *args, **kwargs):
        """
        Plots the trajectory using :mod:`matplotlib.pyplot`.

        :param args: arguments to pass to :func:`~matplotlib.pyplot.plot`
        :param kwargs: keyword arguments to pass to :func:`~matplotlib.pyplot.plot`
        :return: the result of the :func:`matplotlib.pyplot.plot` function.
        """
        # Get label from the kwargs provided, or use self.description as default
        label = kwargs.pop('label', self.description)
        return plt.plot(self.timepoints, self.values, *args, label=label, **kwargs)


def validate_problem(problem):

    problem.validate()

    if problem.method == "MEA":
        moments = filter(lambda x: isinstance(x, Moment), problem.ordered_descriptions)
        if problem.left_hand_side.rows != len(moments):
            raise ValueError("There are {0} equations and {1} moments. "
                             "For MEA problems, the same number is expected.".format(problem.left_hand_side.rows,
                                                                                     len(moments)))
    elif problem.method == 'LNA':
        # FIXME: do some validation for LNA here
        pass


class Simulation(object):
    __problem = None
    __postprocessing = None

    def __init__(self, problem):
        """
        Initialise the simulator object for a given problem
        :param problem:
        :type problem: ODEProblem
        :return:
        """
        self.__problem = problem
        validate_problem(problem)

        if problem.method == 'LNA':
            self.__postprocessing = _postprocess_lna_simulation
        else:
            self.__postprocessing = _postprocess_default

    def _create_cvode_solver(self, initial_constants, initial_values, initial_timepoint=0.0):
        """
        Creates an instance of `CVode` that will be used to simulate the ODEs.

        :param initial_constants: initial values for constants
        :param initial_values: initial values for variables
        :param initial_timepoint: initial timepoint
        :return: instance of `CVode` solver
        :rtype: CVode
        """
        rhs = self.problem.right_hand_side_as_function(initial_constants)
        model = Explicit_Problem(lambda t, x: rhs(x), initial_values, initial_timepoint)
        solver = CVode(model)

        solver.verbosity = 50  # Verbosity flag suppresses output

        # TODO: Make these customisable
        solver.iter = 'Newton'
        solver.discr = 'BDF'
        solver.atol = ATOL
        solver.rtol = RTOL
        solver.linear_solver = 'dense'

        return solver

    def simulate_system(self, initial_constants, initial_values, timepoints):
        """
        Simulates the system for each of the timepoints, starting at initial_constants and initial_values values
        :param initial_constants: list of the initial values for the constants in the model. Must be in the same order
        as in the model
        :param initial_values: List of the initial values for the equations in the problem. Must be in the same order as
        these equations occur. If not all values specified, the remaining ones will be assumed to be 0.
        :param timepoints: A list of time points to simulate the system for
        :return:
        """

        # If not all intial conditions specified, append zeros to them
        # TODO: is this really the best way to do this?
        if len(initial_values) < self.problem.number_of_equations:
            initial_values = np.concatenate((initial_values,
                                             [0.0] * (self.problem.number_of_equations - len(initial_values))))

        initial_timepoint = timepoints[0]
        last_timepoint = timepoints[-1]

        solver = self._create_cvode_solver(initial_constants, initial_values, initial_timepoint)
        try:
            simulated_timepoints, simulated_values = solver.simulate(last_timepoint, ncp_list=timepoints)
        except CVodeError as e:
            # assimulo masks the error that occurs in RHS function
            # by it's CVodeError exception
            # Let's try to call that function ourselves and see if we could cause that error
            # and not mask it
            try:
                self.problem.right_hand_side_as_function(initial_constants)(initial_values)
            except:
                raise
            else:
                # If the right_hand_side_as_function above did not raise any exceptions, re-raise CVode error
                raise e

        trajectories = self.__postprocessing(self.problem, simulated_values, simulated_timepoints)

        return trajectories

    @property
    def problem(self):
        return self.__problem

def _postprocess_default(problem, simulated_values, timepoints):

    trajectories = []
    descriptions = problem.ordered_descriptions

    number_of_timepoints, number_of_simulated_values = simulated_values.shape

    assert(len(descriptions) == number_of_simulated_values)
    assert(len(timepoints) == number_of_timepoints)

    for description, simulated_value_column in zip(descriptions, simulated_values.T):
        trajectories.append(Trajectory(timepoints, simulated_value_column, description))

    return trajectories

def _postprocess_lna_simulation(problem, simulated_values, timepoints):

    # TODO: this should be going through the descriptions of LHS fields, rather than number of species
    # Would make code cleaner

    number_of_species = problem.number_of_species
    # Answer_buffer
    answer = []

    mu_t = np.zeros((len(timepoints), number_of_species), dtype=NP_FLOATING_POINT_PRECISION)
    for species_id in range(number_of_species):

        mu_i = np.zeros(len(timepoints), dtype=NP_FLOATING_POINT_PRECISION)
        for timepoint_index in range(len(timepoints)):

            # For each timepoint
            if species_id == 0:
                # Construct a covariance matrix out of the covariance terms in the model
                V = Matrix(number_of_species, number_of_species, lambda k, l: 0)
                # FIXME: Does the hardcoded 2 really work here?
                # shouldn't it be number_of_species**2
                for v in range(2 * number_of_species):
                    V[v] = simulated_values[timepoint_index, v + number_of_species]

                # Sample new values for each species from a multivariate normal
                mu_t[timepoint_index] = np.random.multivariate_normal(simulated_values[timepoint_index, 0:number_of_species], V)

            mu_i[timepoint_index] = mu_t[timepoint_index][species_id]

        # TODO: some zipping action should be there below to get the description in a nicer way
        trajectory = Trajectory(timepoints, mu_i, problem.ordered_descriptions[species_id])
        answer.append(trajectory)
    return answer


def print_output(output_file, trajectories, initial_conditions, number_of_species, param, timepoints, max_order=None):
    # Check maximum order of moments to output to file/plot
    # TODO: change wherever maxorder comes from for it to be "None" not false.

    # write results to output file (Input file name, parameters,
    # initial conditions, data needed for maximum entropy
    # (no.timepoints, no.species, max order of moments),
    # timepoints, and trajectories for each moment)
    output = open(output_file, 'w')
    try:
        output.write('\n>Parameters: {0!r}\n>Starting values: {1}\n'.format([round(x, 6) for x in param],
                                                                            [round(y, 6) for y in initial_conditions]))

        output.write('#\t{0}\t{1}\t{2}\n'.format(len(timepoints), number_of_species, max_order))
        output.write('time\t{0}\n'.format('\t'.join(map(str, timepoints))))

        # write trajectories of moments (up to maxorder) to output file
        for trajectory in trajectories:
            term = trajectory.description
            if not isinstance(term, Moment):
                continue
            if max_order is None or term.order <= max_order:
                output.write('{0}\t{1}\n'.format(term, '\t'.join(map(str, trajectory.values))))
    finally:
        output.close()

def simulate(problem, trajout, timepoints, initial_constants, initial_variables, maxorder):
    """
    :param simulation_type: either "MEA" or "LNA"
    :param problem: Parsed problem to simulate
    :type problem: ODEProblem
    :param trajout: Name of output file for this function (where simulated trajectories would be stored, i.e. --simout)
    :param timepoints: List of timepoints
    :param initial_constants: List of kinetic parameters
    :param initial_variables: List of initial conditions for each moment (in timeparameters file)
    :param maxorder: Maximum order of moments to output to either plot or datafile. (Defaults to maximum order of moments)
    :return: a list of trajectories resulting from simulation
    """

    # Get required info from the expansion output

    number_of_species = problem.number_of_species

    term_descriptions = problem.ordered_descriptions

    initial_variables = np.array(initial_variables, dtype=NP_FLOATING_POINT_PRECISION)
    initial_constants = np.array(initial_constants, dtype=NP_FLOATING_POINT_PRECISION)
    simulator = Simulation(problem)
    trajectories = simulator.simulate_system(initial_constants, initial_variables, timepoints)

    print_output(trajout, trajectories, initial_variables, number_of_species,
                 initial_constants, timepoints, maxorder)

    return trajectories, term_descriptions

def graphbuilder(soln,momexpout,title,t,momlist):
    """
    Creates a plot of the solutions

    :param soln: output from CVODE (array of solutions at each time point for each moment)
    :param momexpout:
    :param title:
    :param t:
    :param momlist:
    :return:
    """
    simtype = 'momexp'
    LHSfile = open(momexpout)
    lines = LHSfile.readlines()
    LHSindex = lines.index('LHS:\n')
    cindex = lines.index('Constants:\n')
    LHS = []
    for i in range(LHSindex+1,cindex-1):
        LHS.append(lines[i].strip())
        if lines[i].startswith('V'):simtype='LNA'
    fig = plt.figure()
    count = -1
    for el in LHS:
        if '_' in el:
            count+=1
    n = np.floor(np.sqrt(len(LHS)+1-count))+1
    m = np.floor((len(LHS)+1-count)/n)+1
    if n>3:
        n=3
        m=3
    for i in range(len(LHS)):
        if simtype == 'LNA':
            if 'y' in LHS[i]:
                ax = plt.subplot(111)
                ax.plot(t,soln[:][i],label=LHS[i])
                plt.xlabel('Time')
                plt.ylabel('Means')
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          fancybox=True, shadow=True, ncol=3)
        elif simtype == 'momexp':
            if i<(count+9):
                if '_' in LHS[i]:
                    ax1 = plt.subplot(n,m,1)
                    ax1.plot(t,soln[:,i],label=LHS[i])
                    plt.xlabel('Time')
                    plt.ylabel('Means')
                    ax1.legend(loc='upper center', bbox_to_anchor=(0.9, 1.0),
          fancybox=True, shadow=True,prop={'size':12})
                else:
                    ax = plt.subplot(n,m,i+1-count)
                    ax.plot(t,soln[:,i])
                    plt.xlabel('Time')
                    plt.ylabel('['+str(momlist[i])+']')


    fig.suptitle(title)
    plt.show()

