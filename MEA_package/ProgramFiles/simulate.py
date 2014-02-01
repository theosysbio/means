"""
Simulates data for given model, moments, parameters, initial conditions
and method (moment expansion or LNA)
"""
from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import CVode
import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix

# These are the default values in solver.c but they seem very low
RTOL = 1e-4
ATOL = 1e-4
NP_FLOATING_POINT_PRECISION = np.double

class Simulation(object):
    __problem = None

    def __init__(self, problem):
        """
        Initialise the simulator object for a given problem
        :param problem:
        :type problem: ODEProblem
        :return:
        """
        self.__problem = problem

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
        model = Explicit_Problem(rhs, initial_values, initial_timepoint)
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
        initial_timepoint = timepoints[0]
        last_timepoint = timepoints[-1]

        solver = self._create_cvode_solver(initial_constants, initial_values, initial_timepoint)
        simulated_timepoints, simulated_values = solver.simulate(last_timepoint, ncp_list=timepoints)

        return simulated_timepoints, simulated_values

    @property
    def problem(self):
        return self.__problem

def simulate_lna(soln, number_of_species, timepoints):

    # Answer_buffer
    answer = np.zeros((number_of_species, len(timepoints)), dtype=NP_FLOATING_POINT_PRECISION)

    mu_t = np.zeros((len(timepoints), number_of_species), dtype=NP_FLOATING_POINT_PRECISION)
    for species in range(0, number_of_species):

        mu_i = np.zeros(len(timepoints), dtype=NP_FLOATING_POINT_PRECISION)
        for timepoint_index in range(len(timepoints)):

            # For each timepoint
            if species == 0:
                # Construct a covariance matrix out of the covariance terms in the model
                V = Matrix(number_of_species, number_of_species, lambda k, l: 0)
                # TODO: Does the hardcoded 2 really work here?
                # shouldn't it be number_of_species**2
                for v in range(2 * number_of_species):
                    V[v] = soln[timepoint_index, v + number_of_species]
                print V
                print

                # Sample new values for each species from a multivariate normal
                mu_t[timepoint_index] = np.random.multivariate_normal(soln[timepoint_index, 0:number_of_species], V)

            mu_i[timepoint_index] = mu_t[timepoint_index][species]
        answer[species] = mu_i
    return answer


def output_lna_result(initial_conditions, moment_list, mu, number_of_species, param, t,
                      trajout):
    output = open(trajout, 'w')

    try:
        output.write('\n>Parameters: {0!r}\n>Starting values: {1}\n'.format(
            [round(x, 6) for x in param], [round(y, 6) for y in initial_conditions]))
        output.write('time')
        for i in range(0, len(t)):
            output.write('\t' + str(t[i]))
        output.write('\n')
        for m in range(0, number_of_species):
            output.write(', '.join(map(str, moment_list[m])))
            for i in range(0, len(t)):
                output.write('\t' + str(mu[m][i]))
            output.write('\n')
    finally:
        output.close()


def print_mea_output(initial_conditions, maxorder, moment_list, mu, number_of_species, param,
                     t, trajout):
    # Check maximum order of moments to output to file/plot
    if maxorder == False:
        maxorder = max(map(sum, moment_list))

    # Create list of moments as lists of integers
    # (moment_list is list of strings)
    moment_list_int = moment_list[:]

    # write results to output file (Input file name, parameters,
    # initial conditions, data needed for maximum entropy
    # (no.timepoints, no.species, max order of moments),
    # timepoints, and trajectories for each moment)
    output = open(trajout, 'w')
    initcond_str = ''
    for i in initial_conditions:
        initcond_str += (str(i) + ',')
    initcond_str = '[' + initcond_str.rstrip(',') + ']'
    output.write('>\tParameters: ' + str(
        param) + '\n>\tStarting values: ' + initcond_str + '\n')
    output.write('#\t' + str(len(t)) + '\t' + str(number_of_species) + '\t' + str(maxorder) + '\n')
    output.write('time')
    for i in range(0, len(t)):
        output.write('\t' + str(t[i]))
    output.write('\n')
    # write trajectories of moments (up to maxorder) to output file
    for m in range(0, len(moment_list_int)):
        if sum(moment_list_int[m]) <= int(maxorder):
            output.write(', '.join(map(str, moment_list[m])))
            for i in range(0, len(t)):
                output.write('\t' + str(mu[m][i]))
            output.write('\n')
    output.close()

def simulate(simulation_type, problem, trajout, timepoints, initial_constants, initial_variables, maxorder):
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
    lhs = problem.left_hand_side
    moment_list = problem.ordered_moments

    # If not all intial conditions specified, append zeros to them
    initial_variables = initial_variables[:]  # Make a copy before do
    if len(initial_variables) < len(lhs):
        initial_variables.extend([0.0] * (len(lhs) - len(initial_variables)))

    initial_variables = np.array(initial_variables, dtype=NP_FLOATING_POINT_PRECISION)
    initial_constants = np.array(initial_constants, dtype=NP_FLOATING_POINT_PRECISION)
    simulator = Simulation(problem)
    simulated_timepoints, simulation = simulator.simulate_system(initial_constants, initial_variables, timepoints)

    # Interpret the simulation results
    if simulation_type == 'LNA':
        # LNA results build a multivariate gaussian model, which is sampled from here:
        mu = simulate_lna(simulation, number_of_species, timepoints)
        output_lna_result(initial_variables, moment_list, mu, number_of_species, initial_constants, simulated_timepoints,
                          trajout)
        return [simulated_timepoints, mu, moment_list]
    elif simulation_type == 'MEA':
        mu = [simulation[:,i] for i in range(0, len(initial_variables))]
        print_mea_output(initial_variables, maxorder, moment_list, mu, number_of_species,
                         initial_constants, simulated_timepoints, trajout)

        return simulated_timepoints, simulation, moment_list
    


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

