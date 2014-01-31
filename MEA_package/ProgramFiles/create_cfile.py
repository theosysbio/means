############################################################################
# creates C file from template for current model/number of moments
#
# Required inputs (arguments passed to function):
#
# inputfile = input data file (MFK output)
# outputfile = name of cfile (w/o filetype extension)
# t = list of timepoints#
# sd_1 = location of sundials
# sd_2 = location of sundials libraries
##########################################################################

import re
import os
import numpy as np
from powers_py2c import replace_powers
from ode_problem import parse_model

def build_c_library(constants, lhs_equations, ntimepoints, number_of_constants, number_of_equations, outputfile,
                    rhs_equations, sd_1, sd_2, starttime, timestep):
    lhs_equations_list = list(lhs_equations)
    ith = []
    # TODO: Not entirely sure what is happening here what is Ith(), and what is d and why are we adding them
    for i in range(len(lhs_equations)):
        # The Ith is a shorthand for `NV_Ith_S` documented in (http://computation.llnl.gov/casc/sundials/documentation/kin_guide/node7.html),
        # which access to the individual components of the data array of an N_Vector.
        ith.append('{0} = Ith(y, {1});'.format(lhs_equations[i], i + 1))
        lhs_equations_list.append('d{0}'.format(lhs_equations[i]))
    Ithstring = '\n'
    count = 0
    tri = 0
    while count < len(ith):
        if tri == 0:
            Iths = '    ' + ith[count]
            tri += 1
            count += 1
            if count == len(ith):
                Ithstring += Iths + '\n'
        elif tri == 1:
            Iths += ' ' + ith[count]
            tri += 1
            count += 1
            if count == len(ith):
                Ithstring += Iths + '\n'
        elif tri == 2:
            Iths += ' ' + ith[count]
            tri = 0
            count += 1
            Ithstring += Iths + '\n'
    lhs_string = ', '.join(map(str, lhs_equations_list))
    #Create equations
    eqns = []
    for i in range(len(ith), len(lhs_equations_list)):
        eqns.append("{0} = Ith(ydot, {1}) = {2};".format(lhs_equations_list[i],
                                                         i - len(ith) + 1, rhs_equations[i - len(ith)]))

    #Replace 'x**y' with 'pow(x,y)'
    for i in range(len(eqns)):
        eqns[i] = replace_powers(eqns[i])
    for i in range(len(eqns)):
        for j in range(len(constants) - 1, -1, -1):
            eqns[i] = str(eqns[i]).replace(str(constants[j]), 'data->parameters[' + str(j) + ']')
    eqns_string = eqns[0]
    for i in range(1, len(eqns)):
        eqns_string += '\n    ' + eqns[i]

    #Print output
    print_output = 'realtype t'
    for i in range(len(lhs_equations_list) / 2):
        print_output = print_output + ', realtype ' + str(lhs_equations_list[i])

    #SUNDIALS_EXTENDED_PRECISION
    sundials_extended_precision_string = '%14.6Le'
    for i in range((len(lhs_equations_list) / 2) - 1):
        sundials_extended_precision_string += '  %14.6Le'
    sundials_extended_precision_string2 = 't'
    for i in range(len(lhs_equations_list) / 2):
        sundials_extended_precision_string2 += ', ' + str(lhs_equations_list[i])

    #SUNDIALS_DOUBLE_PRECISION
    SDPstring = '%14.6le'
    for i in range((len(lhs_equations_list) / 2) - 1):
        SDPstring += '  %14.6le'

    #last line!
    last = '%14.6e'
    for i in range((len(lhs_equations_list) / 2) - 1):
        last += '  %14.6e'

    #Create input for c file
    template = open('c_template.c')
    lines = template.readlines()
    for i in range(len(lines)):
        if '<NEQ>' in lines[i]:
            lines[i] = lines[i].replace('<NEQ>', str(number_of_equations))
        if '<T0>' in lines[i]:
            lines[i] = lines[i].replace('<T0>', str(starttime))
        if '<T1>' in lines[i]:
            lines[i] = lines[i].replace('<T1>', str(timestep))
        if '<NOUT>' in lines[i]:
            lines[i] = lines[i].replace('<NOUT>', str(ntimepoints))
        if '<NPAR>' in lines[i]:
            lines[i] = lines[i].replace('<NPAR>', str(number_of_constants))
        if '<printoutput>' in lines[i]:
            lines[i] = lines[i].replace('<printoutput>', str(print_output))
        if '<LHS>' in lines[i]:
            lines[i] = lines[i].replace('<LHS>', lhs_string)
        if '<Ith>' in lines[i]:
            lines[i] = lines[i].replace('<Ith>', Ithstring)
        if '<eqns>' in lines[i]:
            lines[i] = lines[i].replace('<eqns>', eqns_string)
        if '<SEP>' in lines[i]:
            lines[i] = lines[i].replace('<SEP>', sundials_extended_precision_string)
        if '<SEP2>' in lines[i]:
            lines[i] = lines[i].replace('<SEP2>', sundials_extended_precision_string2)
        if '<SDP>' in lines[i]:
            lines[i] = lines[i].replace('<SDP>', SDPstring)
        if '<last>' in lines[i]:
            lines[i] = lines[i].replace('<last>', last)

    #Write the C file
    output = open(outputfile + '.c', 'w')
    for line in lines:
        output.write(line)
    output.close()
    # Compile the C file
    #    os.system('g++ -I/cluster/soft/Linux_2.6_64/include -g -O2 -fPIC -c ./'+outputfile+'.c -o '+outputfile+'.o')
    #    os.system('g++ -shared -Wl -o '+outputfile+'.so.1.0 ./'+outputfile+'.o -g -O2 -fPIC /cluster/soft/Linux_2.6_64/lib/libsundials_cvode.a /cluster/soft/Linux_2.6_64/lib/libsundials_nvecserial.a -lm')
    os.system('g++ -I' + sd_1 + ' -g -O2 -fPIC -c ./' + outputfile + '.c -o ' + outputfile + '.o')
    os.system(
        'g++ -shared -o ' + outputfile + '.so.1.0 ./' + outputfile + '.o -g -O2 -fPIC ' + sd_2 + 'libsundials_cvode.a ' + sd_2 + 'libsundials_nvecserial.a -lm')
    return lhs_equations_list


def create_c(inputfile, outputfile, t, sd_1, sd_2):
    """

    :param inputfile: input file - the output of MEA/LNA
    :param outputfile: output file to store the c program (minus .c extension)
    :param t: timepoints returned from `paramtime`
    :param sd_1: location of folder where sundials include files could be found
    :param sd_2: location of folder where sundials lib files could be found
    :return:
    """

    # Timing
    # TODO: By the looks of it these lines ignore the actual timepoint information
    # and assume start time equals zero (otherwise t[1] will not be the timestep)
    # time points are equally distributed
    # TODO: What happens if len(t) = 0
    starttime = float(t[0])
    timestep = float(t[1])
    endtime = float(t[-1])
    ntimepoints = float(len(t))  #NOUT

    ode_problem = parse_model(inputfile)
    constants = ode_problem.constants
    lhs_equations = ode_problem.left_hand_side
    number_of_constants = len(constants)
    number_of_equations = len(lhs_equations)
    rhs_equations = ode_problem.right_hand_side
    # constants, lhs_equations, number_of_constants, number_of_equations, rhs_equations = parse_mea_file(inputfile)

    lhs_equations_list = build_c_library(constants, lhs_equations, ntimepoints, number_of_constants,
                                         number_of_equations, outputfile, rhs_equations, sd_1, sd_2, starttime,
                                         timestep)
    
    return lhs_equations_list
