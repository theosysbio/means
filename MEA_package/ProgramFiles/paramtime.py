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

def paramtime(tpfile,restart, limit):
  
    infile = open(tpfile)
    lines = infile.readlines()

    param = None
    initcond = None
    vary = None
    varyic = None
    if limit==True:
        limits = []
    else:
        limits = None
    

    for i in range(len(lines)):
        if lines[i].startswith('Timepoints:'):
            ts = lines[i+1].rstrip()
            if ts!='':
                ts = ts.split()
                t = [float(point) for point in ts]
            else:
                print "\n  Error:\n  No timepoints entered.\n"
        if lines[i].startswith('Parameters:'):
            params = lines[i+1].rstrip()
            if params!='':
                params = params.split()
                param = [float(p) for p in params]
 
            if restart == True:
                params = lines[i+2].rstrip()
                params = params.split()
                param1 = [float(p) for p in params]
                pranges = []
                for j in range(len(param)):
                    pranges.append((param[j],param1[j]))
                param = pranges

        if lines[i].startswith('Initial'):
            initconds = lines[i+1].rstrip()
            if initconds!='':
                initconds = initconds.split()
                initcond = [float(cond) for cond in initconds]
            
            if restart ==True:
                initconds = lines[i+2].rstrip()
                initconds = initconds.split()
                initcond1 = [float(ic) for ic in initconds]
                icranges = []
                for j in range(len(initcond)):
                    icranges.append((initcond[j], initcond1[j]))
                initcond = icranges

        if lines[i].startswith('Fixed(0)/variable(1) parameters'):
            varys = lines[i+1].rstrip()
            if varys!='':
                varys = varys.split()
                vary = [float(x) for x in varys]
            
        if lines[i].startswith('Fixed(0)/variable(1) initial conditions'):
            if i < len(lines)-1:
                varysic = lines[i+1].rstrip()
                if varysic!='':
                    varysic = varysic.split()
                    varyic = [float(x) for x in varysic]

        if limit ==True:
            if lines[i].startswith('Set parameter limits:'):
                upper_p = param_limits(lines[i+1])
                lower_p = param_limits(lines[i+2])
                for j in range(0,len(upper_p)):
                    limits.append((lower_p[j],upper_p[j]))
            if lines[i].startswith('Set initial conditions limits:'):
                upper_ic = param_limits(lines[i+1])
                lower_ic = param_limits(lines[i+2])
                for k in range(0, len(upper_ic)):
                    limits.append((lower_ic[k], upper_ic[k]))

    return [t,param,initcond,vary, varyic, limits]

