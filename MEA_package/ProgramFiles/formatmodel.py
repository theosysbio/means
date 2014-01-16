import sys
import re

filename = str(sys.argv[1])   #name of data entry file

infile = open(filename)
D = infile.readlines()    #read input data

# Regex to identify relevant sections
nreactions_RE = re.compile('Number of reactions')
nconstants_RE = re.compile('Number of rate constants')
nvariables_RE = re.compile('Number of variables')
stoic_RE = re.compile('Stoichiometry')
s_entry_RE = re.compile('\[(.+)\]')
propensities_RE = re.compile('Reaction propensities')

# Extract required information
for i in range(0,len(D)):
    if nreactions_RE.match(D[i]):
        nreactions = D[i+1].rstrip()
    if nconstants_RE.match(D[i]):
        nrateconstants = D[i+1].rstrip()
    if nvariables_RE.match(D[i]):
        nvariables = D[i+1].rstrip()
    
    if stoic_RE.match(D[i]):
        S = ''
        for j in range(i+1, i+1+int(nvariables)):
            if s_entry_RE.match(D[j]):
                S += str(s_entry_RE.match(D[j]).group(1)) +','
        S = 'Matrix('+nvariables+','+nreactions+',['+S.rstrip(',')+'])'

    if propensities_RE.match(D[i]):
        a = ''
        index = 0
        for k in range(i+1, i+1+int(nreactions)):
            a += '\ta['+str(index)+'] = '+D[k].rstrip() + '\n'
            index+=1
        a = str.replace(a,'y', 'ymat')


output = open('model.py','w')
output.write('from sympy import Matrix\nfrom initialize_parameters import initialize_parameters\n\ndef model():\n\tnreactions = '+nreactions+'\n\tnrateconstants = '+nrateconstants+'\n\tnvariables = '+nvariables+'\n\t[ymat, Mumat, c]=initialize_parameters(nrateconstants,nvariables)\n\tS = '+S+'\n\ta = Matrix(nreactions, 1, lambda i,j:0)\n'+a+'\n\treturn [S, a, nreactions, nvariables, ymat, Mumat, c]')
