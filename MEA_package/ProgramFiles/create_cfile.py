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

    # Read the MEA/LNA output
    file = open(inputfile)
    lines = file.readlines()
    
    # Create important indices for data extraction (find out where data is located)
    MFKindex = lines.index('RHS of equations:\n')
    LHSindex = lines.index('LHS:\n')
    cindex = lines.index('Constants:\n')
    nvarindex = lines.index('Number of variables:\n')

    # Extract MFK (RHS of equations)
    MFK = []
    for i in range(MFKindex+1,LHSindex-1):
        MFK.append(lines[i].rstrip())
    NEQ = len(MFK)      #Number of equations
    
    # Extract constants
    c = []
    for i in range(cindex+1,nvarindex):
        c.append(lines[i].rstrip())
    NPAR = len(c) #Number of parameters

    # Extract LHS of equations
    LHS = []
    Ith = []
    for i in range(LHSindex+1,cindex-1):
        LHS.append(lines[i].rstrip())

    # TODO: Not entirely sure what is happening here what is Ith(), and what is d and why are we adding them
    for i in range(len(LHS)):
        # The Ith is a shorthand for `NV_Ith_S` documented in (http://computation.llnl.gov/casc/sundials/documentation/kin_guide/node7.html),
        # which access to the individual components of the data array of an N_Vector.
        Ith.append(LHS[i]+' = Ith(y,'+str(i+1)+');')
        LHS.append('d'+LHS[i])


    Ithstring = '\n'
    count = 0
    tri = 0
    while count<len(Ith):
        if tri == 0:
            Iths = '    '+Ith[count]
            tri+=1
            count+=1
            if count==len(Ith):
                Ithstring+=Iths+'\n'
        elif tri == 1:
            Iths+=' '+Ith[count]
            tri+=1
            count+=1
            if count==len(Ith):
                Ithstring+=Iths+'\n'
        elif tri == 2:
            Iths+=' '+Ith[count]
            tri = 0
            count+=1
            Ithstring+=Iths+'\n'

    LHSstring = LHS[0]
    for i in range(1,len(LHS)):
        LHSstring+=', '+str(LHS[i])

    #Create equations
    eqns = []
    for i in range(len(Ith),len(LHS)):
        eqns.append(LHS[i]+' = Ith(ydot,'+str(i-len(Ith)+1)+') = '+MFK[i-len(Ith)]+';')

    #Replace 'x**y' with 'pow(x,y)'
    for i in range(len(eqns)):
        eqns[i] = replace_powers(eqns[i])
        
    for i in range(len(eqns)):
        for j in range(len(c)-1,-1,-1):
            eqns[i]=str(eqns[i]).replace(str(c[j]),'data->parameters['+str(j)+']')

    eqnsstring = eqns[0]
    for i in range(1,len(eqns)):
        eqnsstring+='\n    '+eqns[i]

    #Print output
    printoutput = 'realtype t'
    for i in range(len(LHS)/2):
        printoutput = printoutput + ', realtype ' + str(LHS[i])

    #SUNDIALS_EXTENDED_PRECISION
    SEPstring = '%14.6Le'
    for i in range((len(LHS)/2)-1):
        SEPstring+='  %14.6Le'
    SEPstring2 ='t'
    for i in range(len(LHS)/2):
        SEPstring2+=', '+LHS[i]

    #SUNDIALS_DOUBLE_PRECISION
    SDPstring = '%14.6le'
    for i in range((len(LHS)/2)-1):
        SDPstring+='  %14.6le'

    #last line!
    last = '%14.6e'
    for i in range((len(LHS)/2)-1):
        last+='  %14.6e'

    #Create input for c file
    template = open('c_template.c')
    lines = template.readlines()
    for i in range(len(lines)):
        if '<NEQ>' in lines[i]:
            lines[i]=lines[i].replace('<NEQ>',str(NEQ))
        if '<T0>' in lines[i]:
            lines[i]=lines[i].replace('<T0>',str(starttime))
        if '<T1>' in lines[i]:
            lines[i]=lines[i].replace('<T1>',str(timestep))
        if '<NOUT>' in lines[i]:
            lines[i]=lines[i].replace('<NOUT>',str(ntimepoints))
        if '<NPAR>' in lines[i]:
            lines[i]=lines[i].replace('<NPAR>',str(NPAR))
        if '<printoutput>' in lines[i]:
            lines[i]=lines[i].replace('<printoutput>',str(printoutput))
        if '<LHS>' in lines[i]:
            lines[i]=lines[i].replace('<LHS>',LHSstring)
        if '<Ith>' in lines[i]:
            lines[i]=lines[i].replace('<Ith>',Ithstring)
        if '<eqns>' in lines[i]:
            lines[i]=lines[i].replace('<eqns>',eqnsstring)
        if '<SEP>' in lines[i]:
            lines[i]=lines[i].replace('<SEP>',SEPstring)
        if '<SEP2>' in lines[i]:
            lines[i]=lines[i].replace('<SEP2>',SEPstring2)
        if '<SDP>' in lines[i]:
            lines[i]=lines[i].replace('<SDP>',SDPstring)
        if '<last>' in lines[i]:
            lines[i]=lines[i].replace('<last>',last)

    #Write the C file
    output = open(outputfile+'.c','w')
    for line in lines:
        output.write(line)

    output.close()

    # Compile the C file    
#    os.system('g++ -I/cluster/soft/Linux_2.6_64/include -g -O2 -fPIC -c ./'+outputfile+'.c -o '+outputfile+'.o')
#    os.system('g++ -shared -Wl -o '+outputfile+'.so.1.0 ./'+outputfile+'.o -g -O2 -fPIC /cluster/soft/Linux_2.6_64/lib/libsundials_cvode.a /cluster/soft/Linux_2.6_64/lib/libsundials_nvecserial.a -lm')

    os.system('g++ -I' +sd_1+ ' -g -O2 -fPIC -c ./'+outputfile+'.c -o '+outputfile+'.o')
    os.system('g++ -shared -Wl -o '+outputfile+'.so.1.0 ./'+outputfile+'.o -g -O2 -fPIC ' +sd_2+ 'libsundials_cvode.a ' +sd_2+ 'libsundials_nvecserial.a -lm')


    return LHS
