import numpy as np
def param_limits(line):
    limit_strings = line.rstrip()
    limit_strings = limit_strings.split()

    limits = []
    for j in limit_strings:

        try:
            limit = float(j)
        except ValueError:
            if j.strip() == 'N':
                limit = None
            else:
                raise
        limits.append(limit)
    return limits


def paramtime(tpfile, restart, limit, problem):
    infile = open(tpfile)
    lines = infile.readlines()

    param = None
    initcond = None
    vary = None
    varyic = None

    limits_parameters = None
    limits_initial_conditions = None

    if not limit:
        limits_parameters = [None] * len(problem.constants)
        limits_initial_conditions = [None] * len(problem.left_hand_side)
    for i in range(len(lines)):
        if lines[i].startswith('Timepoints:'):
            # Reads timepoints from the parameters file
            # These timepoints must match the timepoints given in the experimental data file used in parameter inference
            ts = lines[i + 1].rstrip()
            if ts != '':
                ts = ts.split()
                t = [float(point) for point in ts]
            else:
                print "\n  Error:\n  No timepoints entered.\n"
        if lines[i].startswith('Parameters:'):
            # Reads the parameters from the file
            # Should be given in order to the symbols used in the model file (i.e. c0, c1, c2, ... etc)

            # Either one line (for simulation/inference from a single starting set of values)
            params = lines[i + 1].rstrip()
            if params != '':
                params = params.split()
                param = map(float, params)

            # Or, if random restarts are used, two lines signifying upper and lower limits for starting values respectively
            # TODO: what happens when two lines given but restart not used?
            # We would take the first line, but maybe we should raise an error?
            if restart:
                params = lines[i + 2].rstrip()
                params = params.split()
                param1 = map(float, params)
                pranges = []
                for j in range(len(param)):
                    pranges.append((param1[j], param[j]))
                param = pranges

                # At this point param is either a list of floats, if --random not used,
                # or a list of tuples containing upper and lower bounds of parameters

        # Read initial conditions for each moment
        # Should be in order they are listed in the output file from moment expansion (i.e. ODEout)
        # under "list of moments". If full set of initial conditions is not provided, any subsequent moments in
        # the list will be set to 0, but values must be given for all moments up to and including the latest one
        # in 'List of moments' that you want to specify
        if lines[i].startswith('Initial'):
            initconds = lines[i + 1].rstrip()
            if initconds != '':
                initconds = initconds.split()
                initcond = [float(cond) for cond in initconds]

            # Similar range handling as in parameters
            if restart == True:
                initconds = lines[i + 2].rstrip()
                initconds = initconds.split()
                initcond1 = [float(ic) for ic in initconds]
                icranges = []
                for j in range(len(initcond)):
                    icranges.append((initcond1[j], initcond[j]))
                initcond = icranges

        # Fixed versus variable moments (1) means variable
        if lines[i].startswith('Fixed(0)/variable(1) parameters'):
            varys = lines[i + 1].rstrip()
            if varys != '':
                varys = varys.split()
                # Not sure why this is converted to float here. TODO: boolean seems more appropriate
                vary = [bool(x) for x in varys]

        # Fixed versus variable initial conditions
        if lines[i].startswith('Fixed(0)/variable(1) initial conditions'):
            if i < len(lines) - 1:
                varysic = lines[i + 1].rstrip()
                if varysic != '':
                    varysic = varysic.split()
                    varyic = [bool(x) for x in varysic]

        # If limit option is set, try reading the parameter limits
        if limit:
            # Set bounds for allowed parameter values if constrained optimisation is used during inference
            # --limit option. Upper and lower bounds are set by the first and second lines respectively, 'N' indicating
            # that a particular bound does not exist.
            if lines[i].startswith('Set parameter limits:'):
                upper_p = param_limits(lines[i + 1])
                lower_p = param_limits(lines[i + 2])
                limits_parameters = zip(lower_p, upper_p)
                # Used to set bounds for allowed initial condition values if running constrained optimisation.
            # Set upper and lower bounds for the allowed ranges as described by 'Set parameter limits'
            if lines[i].startswith('Set initial conditions limits:'):
                upper_ic = param_limits(lines[i + 1])
                lower_ic = param_limits(lines[i + 2])
                limits_initial_conditions = zip(lower_ic, upper_ic)

    if limit:
        if limits_parameters is None:
            raise Exception("Limit option is set, "
                            "but could not read parameter limits from the file {0!r}".format(tpfile))
        if limits_initial_conditions is None:
            raise Exception("Limit option is set, "
                            "but could not read initial condition limits from the file {0!r}".format(tpfile))

    variable_parameters = {}
    for parameter, is_variable, range_ in zip(problem.constants, vary, limits_parameters):
        if not is_variable:
            continue
        variable_parameters[parameter] = range_

    for initial_conditions, is_variable, range_ in zip(problem.left_hand_side, varyic, limits_initial_conditions):
        if not is_variable:
            continue
        variable_parameters[initial_conditions] = range_


    return t, param, initcond, variable_parameters

