"""
Simulates data for given model, moments, parameters, initial conditions
and method (moment expansion or LNA)
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix

from CVODE import CVODE

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

def parse_expansion_output(expansion_output_filename):
    expansion_output_handle = open(expansion_output_filename, 'r')
    try:
        lines = expansion_output_handle.readlines()
    finally:
        expansion_output_handle.close()

    simulation_type = lines[0].rstrip()
    index_of_lhs_in_file = lines.index('LHS:\n')
    index_of_constants_in_file = lines.index('Constants:\n')
    lhs = []
    number_of_species = 0
    for i in range(index_of_lhs_in_file+1,index_of_constants_in_file-1):
        lhs.append(lines[i].strip())
        if lines[i].startswith('y_'):
            number_of_species+=1


    # Get list of moment names/IDs from mfkoutput to create labels for output data file

    momlistindex = lines.index('List of moments:\n')
    moment_list = []
    for i in range(momlistindex+1, len(lines)):
        if lines[i].startswith('['):
            moment_string = str(lines[i].strip('\n[]'))
            moment_list.append(moment_string)

    return simulation_type, number_of_species, lhs, moment_list


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


def output_lna_result(expansion_output_filename, initial_conditions, moment_list, mu, number_of_species, param, t,
                      trajout):
    output = open(trajout, 'w')

    try:
        output.write('>Input file: ' + str(expansion_output_filename) + '\n>Parameters: ' + str(
            param) + '\n>Starting values: ' + str(initial_conditions) + '\n')
        output.write('time')
        for i in range(0, len(t)):
            output.write('\t' + str(t[i]))
        output.write('\n')
        for m in range(0, number_of_species):
            output.write(moment_list[m])
            for i in range(0, len(t)):
                output.write('\t' + str(mu[m][i]))
            output.write('\n')
    finally:
        output.close()


def print_mea_output(expansion_output_filename, initial_conditions, maxorder, moment_list, mu, number_of_species, param,
                     t, trajout):
    # Check maximum order of moments to output to file/plot
    if maxorder == False:
        maxmoment = [int(i) for i in moment_list[-1].split(',')]
        maxorder = sum(maxmoment)

    # Create list of moments as lists of integers
    # (moment_list is list of strings)
    moment_list_int = []
    for i in range(0, len(moment_list)):
        moment_ints = [int(j) for j in moment_list[i].split(',')]
        moment_list_int.append(moment_ints)

    # write results to output file (Input file name, parameters,
    # initial conditions, data needed for maximum entropy
    # (no.timepoints, no.species, max order of moments),
    # timepoints, and trajectories for each moment)
    output = open(trajout, 'w')
    initcond_str = ''
    for i in initial_conditions:
        initcond_str += (str(i) + ',')
    initcond_str = '[' + initcond_str.rstrip(',') + ']'
    output.write('>\tInput file: ' + str(expansion_output_filename) + '\n>\tParameters: ' + str(
        param) + '\n>\tStarting values: ' + initcond_str + '\n')
    output.write('#\t' + str(len(t)) + '\t' + str(number_of_species) + '\t' + str(maxorder) + '\n')
    output.write('time')
    for i in range(0, len(t)):
        output.write('\t' + str(t[i]))
    output.write('\n')
    # write trajectories of moments (up to maxorder) to output file
    for m in range(0, len(moment_list_int)):
        if sum(moment_list_int[m]) <= int(maxorder):
            output.write(moment_list[m])
            for i in range(0, len(t)):
                output.write('\t' + str(mu[m][i]))
            output.write('\n')
    output.close()


def simulate(expansion_output_filename, trajout, lib, t, param, initial_conditions, maxorder):
    """

    :param expansion_output_filename: Name of output file from MFK/LNA (specified by --ODEout)
    :param trajout: Name of output file for this function (where simulated trajectories would be stored, i.e. --simout)
    :param lib: Name of the C file for solver (i.e. --library)
    :param t: List of timepoints
    :param param: List of kinetic parameters
    :param initial_conditions: List of initial conditions for each moment (in timeparameters file)
    :param maxorder: Maximum order of moments to output to either plot or datafile. (Defaults to maximum order of moments)
    :return:
    """

    # Get required info from the expansion output

    # TODO: Replace this with Quentin's code.
    simulation_type, number_of_species, lhs, moment_list = parse_expansion_output(expansion_output_filename)

    # If not all intial conditions specified, append zeros to them
    initial_conditions = initial_conditions[:]  # Make a copy before do
    if len(initial_conditions) < len(lhs):
        initial_conditions.extend([0.0] * (len(lhs) - len(initial_conditions)))

    ####################################################################
    # If simulation type is LNA ...
    ####################################################################


    if simulation_type == 'LNA':
        # solve with selected parameters
        soln = CVODE(lib, t, initial_conditions, param)
        mu = simulate_lna(soln, number_of_species, t)
        output_lna_result(expansion_output_filename, initial_conditions, moment_list, mu, number_of_species, param, t,
                          trajout)

        return [mu, moment_list]

    ####################################################################
    # If simulation type is MEA ...
    ####################################################################
    elif simulation_type == 'MEA':

        # Solve equations for moment trajectories with given parameters

        soln = CVODE(lib, t, initial_conditions, param)
        mu = [soln[:,i] for i in range(0, len(initial_conditions))]

        print_mea_output(expansion_output_filename, initial_conditions, maxorder, moment_list, mu, number_of_species,
                         param, t, trajout)

        return [soln,moment_list]
    


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

