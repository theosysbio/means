"""
Simulates data for given model, moments, parameters, initial conditions
and method (moment expansion or LNA)
"""
from collections import namedtuple
from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import CVode
import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix

# These are the default values in solver.c but they seem very low
from ode_problem import Moment

RTOL = 1e-4
ATOL = 1e-4
NP_FLOATING_POINT_PRECISION = np.double

Trajectory = namedtuple('Trajectory', ['timepoints', 'values', 'description'])



class Simulation(object):
    __problem = None
    __postprocessing = None

    def __init__(self, problem, postprocessing=None):
        """
        Initialise the simulator object for a given problem
        :param problem:
        :type problem: ODEProblem
        :param postprocessing: either None (for no postprocessing),
                               or a string 'LNA' for the sampling required in LNA model
        :return:
        """
        self.__problem = problem

        if postprocessing == 'LNA':
            self.__postprocessing = _postprocess_lna_simulation
        elif postprocessing is None:
            self.__postprocessing = _postprocess_default
        else:
            raise ValueError('Unsupported postprocessing type {0!r}, '
                             'only None and \'LNA\' supported'.format(postprocessing))

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
        :param initial_constants:
        :param initial_values:
        :param timepoints:
        :return:
        """

        # If not all intial conditions specified, append zeros to them
        # TODO: is this really the best way to do this?
        if len(initial_values) < self.problem.number_of_equations:
            initial_values = initial_values[:]  # Make a copy before do
            initial_values.extend([0.0] * (self.problem.number_of_equations - len(initial_values)))

        initial_timepoint = timepoints[0]
        last_timepoint = timepoints[-1]

        solver = self._create_cvode_solver(initial_constants, initial_values, initial_timepoint)
        simulated_timepoints, simulated_values = solver.simulate(last_timepoint, ncp_list=timepoints)

        trajectories = self.__postprocessing(self.problem, simulated_values, simulated_timepoints)

        return simulated_timepoints, trajectories

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
    :return:
    """

    # Get required info from the expansion output

    number_of_species = problem.number_of_species

    term_descriptions = problem.ordered_descriptions
    simulation_type = problem.method

    initial_variables = np.array(initial_variables, dtype=NP_FLOATING_POINT_PRECISION)
    initial_constants = np.array(initial_constants, dtype=NP_FLOATING_POINT_PRECISION)
    simulator = Simulation(problem, 'LNA' if simulation_type == 'LNA' else None)
    simulated_timepoints, trajectories = simulator.simulate_system(initial_constants, initial_variables, timepoints)

    print_output(trajout, trajectories, initial_variables, number_of_species,
                 initial_constants, simulated_timepoints, maxorder)

    return simulated_timepoints, trajectories, term_descriptions

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

