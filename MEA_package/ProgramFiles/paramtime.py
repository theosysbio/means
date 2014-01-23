def param_limits(line):
    upper_p_ = line.rstrip()
    upper_p_ = upper_p_.split()
    upper_p = []
    for j in upper_p_:
        if j.strip() == 'N':
            upper_p.append(j.strip())
        else:
            upper_p.append(float(j))
    return upper_p


def paramtime(tpfile, restart, limit):
    infile = open(tpfile)
    lines = infile.readlines()

    param = None
    initcond = None
    vary = None
    varyic = None
    if limit == True:
        limits = []
    else:
        limits = None

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
                param = [float(p) for p in params]

            # Or, if random restarts are used, two lines signifying upper and lower limits for starting values respectively
            # TODO: what happens when two lines given but restart not used?
            if restart == True:
                params = lines[i + 2].rstrip()
                params = params.split()
                param1 = [float(p) for p in params]
                pranges = []
                for j in range(len(param)):
                    pranges.append((param[j], param1[j]))
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
                    icranges.append((initcond[j], initcond1[j]))
                initcond = icranges

        # Fixed versus variable moments (1) means variable
        if lines[i].startswith('Fixed(0)/variable(1) parameters'):
            varys = lines[i + 1].rstrip()
            if varys != '':
                varys = varys.split()
                # Not sure why this is converted to float here. TODO: boolean seems more appropriate
                vary = [float(x) for x in varys]

        # Fixed versus variable initial conditions
        if lines[i].startswith('Fixed(0)/variable(1) initial conditions'):
            if i < len(lines) - 1:
                varysic = lines[i + 1].rstrip()
                if varysic != '':
                    varysic = varysic.split()
                    varyic = [float(x) for x in varysic]

        # If limit option is set, try reading the parameter limits
        if limit == True:
            # Set bounds for allowed parameter values if constrained optimisation is used during inference
            # --limit option. Upper and lower bounds are set by the first and second lines respectively, 'N' indicating
            # that a particular bound does not exist.
            if lines[i].startswith('Set parameter limits:'):
                upper_p = param_limits(lines[i + 1])
                lower_p = param_limits(lines[i + 2])
                for j in range(0, len(upper_p)):
                    limits.append((lower_p[j], upper_p[j]))
                # Used to set bounds for allowed initial condition values if running constrained optimisation.
            # Set upper and lower bounds for the allowed ranges as described by 'Set parameter limits'
            if lines[i].startswith('Set initial conditions limits:'):
                upper_ic = param_limits(lines[i + 1])
                lower_ic = param_limits(lines[i + 2])
                for k in range(0, len(upper_ic)):
                    limits.append((lower_ic[k], upper_ic[k]))

    return [t, param, initcond, vary, varyic, limits]

