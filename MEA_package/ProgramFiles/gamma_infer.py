"""
Functions used for inference using parametric distribution models

Nelder-Mead simplex algorithm used to minimise negative log likelihood
i.e. -ln L(x|theta) where theta is the set of parameters to vary
(selected kinetic parameters and initial conditions).
Likelihood (L) is calculated by approximating the distribution
by the chosen density function (--pdf=normal, gamma, lognormal), and
assuming independence between timepoints and sample data points.

For each set of parameters (i0), the mean and variance trajectories
are simulated by solving the MFK equations.  These values are used
to sum the log likelihood over each time/data point.
"""
import os
import sys
from math import sqrt, log, pi, lgamma

from scipy.optimize import fmin

from CVODE import CVODE
from sumsq_infer import make_i0, i0_to_test, sample_data


def mv_index(mfkoutput, mom_names):
    """
    Get indices for mean/variance data for each species of CVODE output.
    Return lists `mean_id` and `var_id` where position in list corresponds to species number.
    Also returns the list `sp_id`, which contains species number for each of the experimental datasets/trajectories
    :param mfkoutput:
    :param mom_names:

    """
    mfkfile = open(mfkoutput)
    lines = mfkfile.readlines()
    mfkfile.close()

    # Get list of moments from mfkoutput (i.e. moments returned by CVODE)
    momlistindex = lines.index('List of moments:\n')
    moments_list = []
    for i in range(momlistindex + 1, len(lines)):
        if lines[i].startswith('['):
            moment_str = str(lines[i].strip('\n[]'))
            moment_str_list = moment_str.split(',')
            moment_int_list = [int(p) for p in moment_str_list]
            moments_list.append(moment_int_list)

    # Get indices for mean/variance data within the CVODE results
    mean_id = [moments_list.index(m) for m in moments_list if sum(m) == 1]
    var_id = [0] * len(mean_id)
    for m in moments_list:
        if sum(m) == 2 and 2 in m:
            var_id[m.index(2)] = moments_list.index(m)

    # Get species id for sample data 
    sp_id = []
    for j in range(len(mom_names)):
        # TODO: The following line silently assumes that `sum(mom_names) == 1`, e.g. `mom_names == [0,1,0]` or similar.
        # We need to make this assumption explicit
        sp_id.append(mom_names[j].index(1))

    # Get indices in CVODE solutions for the moments in sample data
    mom_index_list = [moments_list.index(m) for m in mom_names]

    return sp_id, mean_id, var_id, moments_list, mom_index_list


def eval_density(mean, var, x, distribution):
    """
    Calculates gamma/lognormal/normal pdf given mean variance, x
    where x is the experimental species number measured at a particular timepoint. Returns ln(pdf)
    :param mean: mean
    :param var: variance
    :param x: experimental species number measured at a particular timepoint
    :param distribution: distribution to consider. Either 'gamma', 'normal' and 'lognormal'
    :return: normal log of the pdf
    """
    if distribution == 'gamma':
        b = var / mean
        a = mean / b
        logpdf = (a - 1) * log(x) - (x / b) - a * log(b) - lgamma(a)

    elif distribution == 'normal':
        logpdf = -(x - mean) ** 2 / (2 * var) - log(sqrt(2 * pi * var))

    elif distribution == 'lognormal':
        logpdf = -(log(x) - mean) ** 2 / (2 * var) - log(x * sqrt(2 * pi * var))

    return logpdf

def read_maxent_output():
    """
    Function to get output from maximum entropy calculation.
    Returns list of lists of required data for each timepoint (except t0) in format:
    `[[t1, mean, logZ, lambda 1, lambda 2, ..., lambda n], [t2 data..],[t3 data..]]`
    """

    maxent_out = open('maxent_out.txt')  # TODO: Hardcoded filename?
    lines = maxent_out.readlines()
    maxent_data = []
    for i in range(1, len(lines)):
        line = lines[i].strip()
        if line != '':
            line = line.split('\t')
            float_line = [float(j) for j in line]
            maxent_data.append(float_line)
    return maxent_data


def eval_maxent_density(x, maxent_data_t):
    logZ = maxent_data_t[2]
    lambdas = maxent_data_t[3:]
    logpdf = -logZ
    for i in range(0, len(lambdas)):
        logpdf -= (lambdas[i] * (x ** (i + 1)))
    return logpdf


#####################################################################
# Optimise function, used for parameter inference (minimizes
# the distance/cost funtion using a simplex algorithm)
# Arguments: 
# param, vary      kinetic parameters, and list to indicate which to infer
# initcond,varyic  initial conditions, " "
# sample           name of experimental data file
# cfile            name of C library (specified by --library)
# mfkoutput        name of file produced by MFK (specified by --ODEout)
# distribution     parametric model (specified by --pdf)
######################################################################     


def optimise(param, vary, initcond, varyic, limits, sample, cfile, mfkoutput, distribution):
    """
    Optimise function, used for parameter inference (minimizes the distance/cost funtion using a simplex algorithm)
    :param param: kinetic parameters
    :param vary: list which kinetic parameters to infer
    :param initcond: initial conditions
    :param varyic: list of which initial conditions to vary
    :param limits:
    :param sample: name of experimental data file
    :param cfile: name of C library (i.e. `--library` option)
    :param mfkoutput: name of file produced by MFK (specified by `--ODEout`)
    :param distribution: parametric model (specified by `--pdf`)
    :return:
    """
    i0 = make_i0(param, vary, initcond, varyic)    # create initial i0

    # Get required info from MFK output file, extend initcond with default 
    # value of 0 if only some of the initial moment values are specified

    mfkfile = open(mfkoutput)
    lines = mfkfile.readlines()
    mfkfile.close()
    nSpecies_index = lines.index('Number of variables:\n')
    nspecies = int(lines[nSpecies_index + 1])
    nEquationsindex = lines.index('Number of equations:\n')
    nEquations = int(lines[nEquationsindex + 1].strip())
    if len(initcond) != nEquations:
        initcond += ([0] * (nEquations - len(initcond)))

    ######################################################################
    # Evaluates distance (cost) function for current set of values in i0
    #
    # At each iteration, this function is called by fmin and calculated 
    # using the current values in i0.  Returned value (dist) is minimised
    # by varying i0
    #
    # Distance is sum of -log(likelihood) assuming independence between
    # time/data points
    #####################################################################

    def distance(i0, param, vary, initcond, varyic, sp_id, mu, t, cfile, mean_id, var_id, distribution):

        # value returned if parameters or means < 0 or outside limits
        max_dist = 1.0e10

        # parameters to solve for
        (test_param, test_initcond) = i0_to_test(i0, param, vary, initcond, varyic)

        # Check parameters are positive, and within limits (if --limit used)
        if any(i < 0 for i in test_param):     # parameters cannot be negative
            return max_dist
        if any(j < 0 for j in test_initcond[0:nspecies]):
            return max_dist      # disallow negative means

        if limits != None:
            for a in range(0, len(i0)):
                l_limit = limits[a][0]
                u_limit = limits[a][1]
                if l_limit != 'N':
                    if i0[a] < l_limit:
                        return max_dist
                if u_limit != 'N':
                    if i0[a] > u_limit:
                        return max_dist

        # get moment expansion result with current parameters
        test_soln = CVODE(cfile, t, test_initcond, test_param)
        tmu = [test_soln[:, i] for i in range(0, len(initcond))]
        logL = 0

        ##############################################################
        # Max entropy (1D only)
        #############################################################
        if distribution == 'maxent':

            # write 'test' moment results to file to pass to maxent module
            maxent_in = open('maxent_in.txt', 'w')
            maxent_in.write('time')
            for i in range(0, len(t)):
                maxent_in.write('\t' + str(t[i]))
            maxent_in.write('\n')
            for m in range(0, len(initcond)):
                maxent_in.write(str(m + 1))
                for i in range(0, len(t)):
                    maxent_in.write('\t' + str(tmu[m][i]))
                maxent_in.write('\n')
            maxent_in.close()

            # run maximum entropy calculations using 'test' moment trajectories
            os.system('python centralMaxent.py maxent_in.txt maxent_out.txt')

            # read in data for likelihood calculations
            maxent_data = read_maxent_output()

            # calculate likelihood (only uses timepoints 1 onwards)
            for i in range(0, len(sp_id)):
                for j in range(1, len(t)):
                    data_x = mu[i][j]
                    try:
                        logL += eval_maxent_density(data_x, maxent_data[j - 1])
                    except:
                        print "Maximum entropy calculation failed."
                        sys.exit()
            dist = -logL
            y_list.append(dist)
            i0_list.append(i0[0])

            """ (Used during testing of maxent)
            os_text='cp maxent_out.txt maxent_out_'+str(len(y_list))+'.txt'
            os.system(os_text)
	    os.system('cp maxent_in.txt maxent_in_'+str(len(y_list))+'.txt')
            maxent_file = open('maxent_file.txt','a')
            maxent_file.write('\nNo. '+str(len(y_list))+'\tCurrent i0:\t'+str(i0[0])+'\nDistance:\t'+str(dist))
            maxent_file.close()
            """
            return dist



        ###############################################################
        # If a parametric distribution used..
        ################################################################
        for i in range(0, len(sp_id)):
            species_id = sp_id[i]
            for j in range(0, len(t)):    # var at t=0 often set to 0 therefore can't calculate likelihood here
                if mu[i][j] == 'N':
                    logL += 0
                else:
                    xt = mu[i][j]
                    mean = tmu[mean_id[species_id]][j]
                    var = tmu[var_id[species_id]][j]

                    if mean < 0 or var < 0:
                        return max_dist  # mean,var must be positive
                    elif var == 0:   # can't calculate parametric likelihoods for zero variance, so pass
                        logL += 0
                    else:
                        logL += eval_density(mean, var, xt, distribution)

        dist = -logL
        y_list.append(dist)
        i0_list.append(i0[0])
        return dist

    # callback: function called after each iteration (each iteration will involve several
    # distance function evaluations).  Use this to save data after each iteration if wanted.
    # x is the current i0 returned after that iteration.    

    def my_callback(x):
        it_param.append(x[0])
        it_no.append(len(it_param))
        """ Used during testing
        it_dist.append(y_list[-1])
        maxent_file = open('maxent_file.txt','a')
        maxent_file.write('\nCallback\n'+str(x[0])+'\n')
        """

    # create lists to collect data at each iteration (used during testing)
    y_list = []
    i0_list = []
    it_no = []
    it_param = []
    it_dist = []

    # read sample data from file and get indices for mean/variances in CVODE output
    (mu, t, mom_names) = sample_data(sample)
    (sp_id, mean_id, var_id, moments_list, mom_index_list) = mv_index(mfkoutput, mom_names)
    #(mom_index_list,moments_list) = mom_indices(mfkoutput, mom_names)

    # minimise defined distance function, with provided starting parameters
    result = fmin(distance, i0,
                  args=(param, vary, initcond, varyic, sp_id, mu, t, cfile, mean_id, var_id, distribution),
                  ftol=0.000001, disp=0, full_output=True, callback=my_callback)

    return result, mu, t, initcond, mom_index_list, moments_list


