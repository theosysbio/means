"""
Simulates data for given model, moments, parameters, initial conditions
and method (moment expansion or LNA)
"""
from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import CVode
import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix



#######################################################################
# Variables within function:
# moment_list   Contains the names for each moment (e.g. [1,0,1], where
#               numbers are the power that each species is raised to
# soln          Array returned by CVODE, each row is a list of moments/
#               values at a particular timepoint
# mu            List where each item is a timecourse (as list) for a
#               particular moment/value
#
# Simulate function returns (mu, moment_list) i.e. trajectories and 
# corresponding list of names identifying the moments
########################################################################
# These are the default values in solver.c but they seem very low
RTOL = 1e-4
ATOL = 1e-4
NP_FLOATING_POINT_PRECISION = np.double

def simulate_lna(soln, number_of_species, t):

    mu = [0] * number_of_species
    mu_t = [0] * len(t)
    for i in range(0, number_of_species):
        mu_i = [0] * len(t)
        for j in range(len(t)):
            if i == 0:
                V = Matrix(number_of_species, number_of_species, lambda k, l: 0)
                for v in range(2 * number_of_species):
                    V[v] = soln[j, v + number_of_species]
                mu_t[j] = np.random.multivariate_normal(soln[j, 0:number_of_species], V)
            mu_i[j] = mu_t[j][i]
        mu[i] = mu_i

    # write results to output file (Input file name, parameters, initial condtions,
    # timepoints, and trajectories for each moment)
    return mu


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


def rhs_factory(rhs_function, constant_values):

    def rhs(t, variable_values):
        """
        Computes the values for right-hand-sides of the equation, used to define model
        """
        all_values = np.concatenate((constant_values, variable_values))
        return rhs_function(*all_values)

    return rhs

def simulate_system(rhs, initial_values, timepoints):
    initial_timepoint = timepoints[0]

    model = Explicit_Problem(rhs, initial_values, initial_timepoint)
    solver = CVode(model)
    solver.verbosity = 50  # Verbosity flag suppresses output
    solver.iter = 'Newton'
    solver.discr = 'BDF'
    solver.atol = ATOL
    solver.rtol = RTOL
    solver.linear_solver = 'dense'

    number_of_timesteps = len(timepoints)
    simulated_timepoints = np.empty(number_of_timesteps, dtype=NP_FLOATING_POINT_PRECISION)
    simulated_values = np.empty((number_of_timesteps,
                                 len(initial_values)), dtype=NP_FLOATING_POINT_PRECISION)
    simulated_timepoints, simulated_values = solver.simulate(timepoints[-1], ncp_list=timepoints)

    return simulated_timepoints, simulated_values


def simulate(simulation_type, problem, trajout, lib, timepoints, initial_constants, initial_variables, maxorder):
    """
    :param simulation_type: either "MEA" or "LNA"
    :param problem: Parsed problem to simulate
    :type problem: ODEProblem
    :param trajout: Name of output file for this function (where simulated trajectories would be stored, i.e. --simout)
    :param lib: Name of the C file for solver (i.e. --library)
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
    rhs_function = rhs_factory(problem.rhs_as_function, initial_constants)

    simulated_timepoints, simulation = simulate_system(rhs_function, initial_variables, timepoints)

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

