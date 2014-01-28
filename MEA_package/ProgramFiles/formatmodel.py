import sys
import re

# Regex to identify relevant sections
REGEXP_NREACTIONS = re.compile('Number of reactions')
REGEXP_NCONSTANTS = re.compile('Number of rate constants')
REGEXP_NVARIABLES = re.compile('Number of variables')
REGEXP_STOICHIOMETRY = re.compile('Stoichiometry')
REGEXP_S_ENTRY = re.compile('\[(.+)\]')
REGEXP_PROPENSITIES = re.compile('Reaction propensities')

OUTPUT_FILE = 'model.py'

def parse_model(input_filename, output_file):

    infile = open(input_filename)
    try:
        D = infile.readlines()    #read input data
    finally:
        infile.close()

    # Extract required information
    for i in range(0,len(D)):
        if REGEXP_NREACTIONS.match(D[i]):
            nreactions = D[i+1].rstrip()
        if REGEXP_NCONSTANTS.match(D[i]):
            nrateconstants = D[i+1].rstrip()
        if REGEXP_NVARIABLES.match(D[i]):
            nvariables = D[i+1].rstrip()

        if REGEXP_STOICHIOMETRY.match(D[i]):
            S = ''
            for j in range(i+1, i+1+int(nvariables)):
                if REGEXP_S_ENTRY.match(D[j]):
                    S += str(REGEXP_S_ENTRY.match(D[j]).group(1)) +','
            S = 'Matrix('+nvariables+','+nreactions+',['+S.rstrip(',')+'])'

        if REGEXP_PROPENSITIES.match(D[i]):
            a = ''
            index = 0
            for k in range(i+1, i+1+int(nreactions)):
                a += '\ta['+str(index)+'] = '+D[k].rstrip() + '\n'
                index+=1
            a = str.replace(a,'y', 'ymat')


    output = open(output_file,'w')
    try:
        output.write('from sympy import Matrix\nfrom initialize_parameters import initialize_parameters\n\ndef model():\n\tnreactions = '+nreactions+'\n\tnrateconstants = '+nrateconstants+'\n\tnvariables = '+nvariables+'\n\t[ymat, Mumat, c]=initialize_parameters(nrateconstants,nvariables)\n\tS = '+S+'\n\ta = Matrix(nreactions, 1, lambda i,j:0)\n'+a+'\n\treturn [S, a, nreactions, nvariables, ymat, Mumat, c]')
    finally:
        output.close()

if __name__ == '__main__':
    parse_model(sys.argv[1], OUTPUT_FILE)