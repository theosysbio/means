import os
import sys
from ode_problem import parse_problem
from paramtime import paramtime
from simulate import simulate, graphbuilder
from sumsq_infer import optimise, write_inference_results, graph
from hypercube import hypercube
import gamma_infer

def printOptions():
    print "\nList of possible options:"

    print "\n House-keeping:"
    print "  --wd\t\tSpecify the working directory. This will contain all input and\n\t\toutput files. Default Inoutput folder provided."

    print "\n Moment expansion approximation:"
    print "  --MEA\t\tCreate a system of ODEs using moment expansion from the model\n\t\tspecified with --model."
    print "  --model\tSpecify the model input file. Use format in modeltemplate.txt.\n\t\tE.g. --model=yourmodel.txt."
    print "  --nMom\tNumber of moments used in expansion. Default --nMom=2."
    print "  --ODEout\tName of output file. Default --ODEout=ODEout."

    print "\n Linear noise approximation:"
    print "  --LNA\t\tCreate a system of ODEs using LNA. Use --model and --ODEout\n\t\toptions as above."

    print "\n Compiling the ODE solver:"
    print "  --compile\tCreate and compile the file needed for the ODE solver, use\n\t\t--ODEout and --timeparam to specify model and timepoints."
    print "  --library\tSpecify name for the C library with no file type extension.\n\t\tDefault --library=solver."
    print "  --timeparam\tName of file containing timepoints using format in\n\t\tparamtimetemp.txt. Required for --compile. Later used to input\n\t\tother parameters for inference or simulation."
    print "  --sd1\t\tPath to directory containing sundials header files.\n\t\tDefault --sd1=/cluster/soft/Linux_2.6_64/include/."
    print "  --sd2\t\tPath to directory containing sundials libraries.\n\t\tDefault --sd2=/cluster/soft/Linux_2.6_64/lib/."

    print "\n Simulation:"
    print "  --sim\t\tSimulate moment trajectories for a given set of parameters.\n\t\tUse --library, --timeparam and --ODEout to specify required\n\t\tinformation."
    print "  --simout\tSpecify filename for simulated trajectory data output."
    print "  --maxorder\tSpecify the maximum order of moments to simulate (only for\n\t\t--MEA).  Default = maxorder of MEA model used"



    print "\n Parameter inference:"
    print "  --infer\tInfer model parameters using experimental data.\n\t\tUse --timeparam and --data to provide required information."
    print "  --data\tSpecify experimental data file to be used for parameter\n\t\tinference. Timepoints must be the same in both --data and\n\t\t--timeparam files."
    print "  --inferfile\tName of parameter inference output file.\n\t\tDefault --inferfile=inference.txt."
    print "  --restart\tUse Latin Hypercube Sampling for random restarts. Use\n\t\t--timeparam to specify ranges for both kinetic parameters and\n\t\tinitial conditions. For fixed starting parameter values, enter\n\t\tsame value for upper and lower bound."
    print "  --nRestart\tSpecify the number of random restarts. Default --nRestart=4."
    print "  --limit\tConstrain parameter values during optimisation. Use --timeparam\n\t\tto set constraints."
    print "  --pdf\t\tChoose the probability density function used to approximate\n\t\tlikelihood for each species/timepoint.\n\t\tOptions: gamma, normal, lognormal."
    print "  --maxent\tUse maximum entropy to approximate probability density."

    print "\n Graph options:"
    print "  --plot\tPlot simulated or inferred moment trajectories."
    print "  --plottitle\tSpecify plot title."

    print "\n  --help\tPrints out this list of options.\n"

def run():
    
    MFK = False
    model = False
    nMoments = 2
    ODEout = 'ODEout'
    createcfile = False
    library = 'solver'
    tpfile = None
    solve = False
    maxorder = None
    plot = False
    plottitle = ''
    trajout = 'traj.txt'
    infer = False
    inferfile = 'inference.txt'
    exptdata = None
    restart = False
    nRestart=4
    limit = False
    wd = '../Inoutput/'
    distribution = False
    LNA = False
    sundials_1 = '/cluster/soft/Linux_2.6_64/include/'
    sundials_2 = '/cluster/soft/Linux_2.6_64/lib/'


    for i in range(1,len(sys.argv)):
        if sys.argv[i].startswith('--'):
            option = sys.argv[i][2:]

            if option == 'help':
                printOptions()
                sys.exit()
            elif option == 'MEA': MFK = True
            elif option[0:6] == 'model=':model = option[6:]
            elif option[0:5] == 'nMom=':nMoments = option[5:]
            elif option[0:7] == 'ODEout=':ODEout = option[7:]
            elif option == 'compile' : createcfile = True # TODO: this is not used any more, the only reason we keep this
                                                          # is because I do nt want to change all regression tests just now
            elif option[0:8] == 'library=':library = option[8:]
            elif option[0:10]=='timeparam=':tpfile=option[10:]
            elif option[0:4]=='sd1=':sundials_1=option[4:]
            elif option[0:4]=='sd2=':sundials_2=option[4:]
            elif option == 'sim' : solve = True
            elif option == 'plot' : plot = True
            elif option[0:10] == 'plottitle=' : plottitle = option[10:]
            elif option[0:7] == 'simout=' : trajout = option[7:]
            elif option[0:9] == 'maxorder=' : maxorder = int(option[9:])
            elif option == 'infer' : infer = True
            elif option[0:10] == 'inferfile=': inferfile = option[10:]
            elif option[0:5] == 'data=' : exptdata=option[5:]
            elif option == 'restart':restart=True
            elif option[0:9] == 'nRestart=':nRestart=option[9:]
            elif option[0:5] == 'limit':limit=True
            elif option[0:3] == 'wd=': wd = option[3:]
            elif option[0:4] == 'pdf=': distribution = option[4:]
            elif option[0:6] == 'maxent': distribution = 'maxent'
            elif option[0:3] == 'LNA' : LNA = True
            elif option.startswith('random-seed='):
                import random
                random_seed = int(option[12:])
                print 'Setting random seed to {0}'.format(random_seed)
                random.seed(random_seed)
                import numpy.random
                numpy.random.seed(random_seed)
            elif not(sys.argv[i-1][2:] == 'LNA'):
                print "\nunknown option "+sys.argv[i]
                printOptions()
                sys.exit()
        elif not(sys.argv[i-1][2:]=='LNA'):
            print "\nunknown option "+sys.argv[i]
            printOptions()
            sys.exit()

    if (MFK == True) and (LNA == True):
        print "\n  Error:\n  Please choose EITHER --MEA or --LNA.\n"
        sys.exit()

    if MFK == True:
        if model == False:
            print "\n No input model file given for moment expansion.\n Try:\n\t--model=modelname.txt\n"
            sys.exit()
        else:
            if os.path.exists(wd+model)==False:
                print "\n  Error:\n  "+model+"  does not exist in working directory.\n  Please try again with correct model filename.\n"
                sys.exit()
            else:
                os.system('python MEA.py '+wd+model+' '+str(nMoments)+' '+wd+ODEout)

    if LNA == True:
        if model == False:
            print "\n No input model file given LNA.\n Try:\n\t--model=modelname.txt\n"
            sys.exit()
        else:
            if os.path.exists(wd+model)==False:
                print "\n  Error:\n  "+model+"  does not exist in working directory.\n  Please try again with correct model filename.\n"
                sys.exit()
            else:
                os.system('python LNA.py '+wd+model+' '+wd+ODEout)

    if solve and infer:
        print "\n  Error:\n  Please choose EITHER --solve or --infer.\n"
        sys.exit()

    if solve:
        if not os.path.exists(wd+tpfile):
            print "\n  Error:\n  "+tpfile+"  does not exist in working directory.\n  Please try again with correct timepoint/parameter filename.\n"
            sys.exit(1)

        [t,param,initcond,vary, varyic, limits] = paramtime(wd+tpfile,restart, limit)
        problem = parse_problem(wd+ODEout)  # TODO: os.path.join

        simulated_timepoints, solution, momlist = simulate(problem,
                                         wd+trajout,t,param,initcond, maxorder)

        if plot:
            graphbuilder(solution,wd+ODEout,plottitle,simulated_timepoints,momlist)

    if infer:
        if not tpfile:
            print "\n No timepoints/parameters/initial conditions given for inference.\n " \
                  "Please provide a file in the format of paramtimetemp.txt."
            sys.exit(1)
        if not os.path.exists(wd+tpfile):
            print "\n  Error:\n  "+tpfile+"  does not exist in working directory.\n  " \
                                          "Please try again with correct " \
                                          "timepoint/parameter/initial conditions filename.\n"
            sys.exit(1)
        if exptdata is None:
            print "\n No experimental data provided for inference.\n " \
                  "Please try again specifying your data file with the --data option."
            sys.exit(1)
        if not os.path.exists(wd+exptdata):
            print "\n  Error:\n  "+exptdata+"  does not exist in working directory.\n  " \
                                            "Please try again with correct experimental data filename.\n"
            sys.exit(1)

        problem = parse_problem(wd+ODEout)
        # If no random restarts selected:
        if not restart:
            [t, param, initcond, vary, varyic, limits] = paramtime(wd + tpfile, restart, limit)

            if not distribution:        # inference using generalised method of moments
                result, t, observed_trajectories, initcond_full = optimise(param, vary, initcond, varyic,
                                                        limits, wd + exptdata,
                                                        problem)
            else:      # Use parametric or maxent distribution to approximate likelihood
                result, t, observed_trajectories, initcond_full = gamma_infer.optimise(
                                                                                                    problem,
                                                                                                    param, vary,
                                                                                                    initcond, varyic,
                                                                                                    limits,
                                                                                                    wd + exptdata,
                                                                                                    distribution)
            restart_results = [[result, None, param, initcond]]

        # Else if random restarts selected
        else:
            try:
                [t, param, initcond, vary, varyic, limits] = paramtime(wd + tpfile, restart, limit)
            except ValueError:
                print '{0} is not in correct format. Ensure you have entered upper and lower bounds ' \
                      'for all parameter values.'.format(tpfile)
                sys.exit(1)
            all_params = hypercube(int(nRestart), param[:] + initcond[:])
            restart_results = []
            for n in all_params:
                param_n = n[0:len(param)]
                initcond_n = n[len(param):]
                # if distance function used for inference
                if not distribution:
                    result, t, observed_trajectories, initcond_full = optimise(param_n, vary, initcond_n,
                                                                                          varyic, limits,
                                                                                          wd + exptdata,
                                                                                          problem)
                # Else if parametric approximation
                else:
                    result, t, observed_trajectories, initcond_full = gamma_infer.optimise(
                        problem,
                        param_n, vary,
                        initcond_n,
                        varyic, limits,
                        wd + exptdata,
                        distribution)
                restart_results.append([result, observed_trajectories, param_n, initcond_n])

            restart_results.sort(key=lambda x: x[0][1], reverse=False)

        # write results to file (default name 'inference.txt') and plot graph if selected
        write_inference_results(restart_results, t, vary, initcond_full, varyic, wd + inferfile)
        if plot:
            graph(problem, restart_results[0], observed_trajectories, t, initcond_full, vary, varyic, plottitle)


run()
