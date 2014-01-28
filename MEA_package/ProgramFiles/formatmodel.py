import sys
import re
from sympyhelpers import to_sympy_matrix, to_sympy_column_matrix, to_list_of_symbols
# Regex to identify relevant sections
import sympy

REGEXP_NREACTIONS = re.compile('Number of reactions')
REGEXP_NCONSTANTS = re.compile('Number of rate constants')
REGEXP_NVARIABLES = re.compile('Number of variables')
REGEXP_STOICHIOMETRY = re.compile('Stoichiometry')
REGEXP_S_ENTRY = re.compile('\[(.+)\]')
REGEXP_PROPENSITIES = re.compile('Reaction propensities')

OUTPUT_FILE = 'model.py'

class Model(object):
    """
    Stores the model of reactions we want to analyse
    """

    # These are private (as indicated by __, the code is a bit messier, but we can ensure immutability this way)
    __constants = None
    __variables = None
    __propensities = None
    __stoichiometry_matrix = None


    def __init__(self, constants, variables, propensities, stoichiometry_matrix):
        """
        Creates a `Model` object that stores the model of reactions we want to analyse
        :param constants: constants of the model, as `sympy` symbols
        :param variables: variables of the model, as `sympy` symbols, i.e. speces
        :param propensities: a matrix of propensities for each of the reaction in the model.
        :param stoichiometry_matrix: stoichiometry matrix for the model
        """
        self.__constants = to_list_of_symbols(constants)
        self.__variables = to_list_of_symbols(variables)
        self.__propensities = to_sympy_column_matrix(to_sympy_matrix(propensities))
        self.__stoichiometry_matrix = to_sympy_matrix(stoichiometry_matrix)

        self.validate()

    def validate(self):
        """
        Validates whether the particular model is created properly
        """
        if self.stoichiometry_matrix.cols != self.propensities.rows:
            raise ValueError('There must be a column in stoichiometry matrix for each row in propensities matrix')

        if self.stoichiometry_matrix.rows != len(self.variables):
            raise ValueError('There must be a row in stoichiometry matrix for each variable')


    # Expose public interface for the specified instance variables
    # Note that all properties here are "getters" only, thus assignment won't work
    @property
    def constants(self):
        return self.__constants

    @property
    def variables(self):
        return self.__variables

    @property
    def propensities(self):
        return self.__propensities

    @property
    def stoichiometry_matrix(self):
        return self.__stoichiometry_matrix

    # Similarly expose some interface for accessing tally counts
    @property
    def number_of_reactions(self):
        return len(self.__propensities)

    @property
    def number_of_variables(self):
        return len(self.__variables)

    @property
    def number_of_constants(self):
        return len(self.__constants)

def initialize_parameters(nrateconstants,nvariables):

    mumat = sympy.Matrix(nvariables, 1, lambda i,j:sympy.var('y_%d' % (i)))
    Mumat = sympy.Matrix(nvariables, 1, lambda i,j:sympy.var('mu_%d' % (i)))
    c = sympy.Matrix(nrateconstants, 1, lambda i,j:sympy.var('c_%d' % (i)))

    return [mumat, Mumat, c]

def index_to_symbol(indexed_string):
    """
    Replaces indexed form to symbols, e.g. would replace c[0] to c_0
    :param indexed_string: string containing several indexed vectors
    :return:
    """
    return re.sub("(\w+)\[(\d+)\]", r"\1_\2", indexed_string)

def format_model_to_legacy_model_py(model):


    model_str = """
from sympy import Matrix, Symbol
from initialize_parameters import initialize_parameters
def model():
    nreactions = {number_of_reactions!r}
    nrateconstants = {number_of_constants!r}
    nvariables = {number_of_species!r}
    [ymat, Mumat, c]=initialize_parameters(nrateconstants,nvariables)
    {stoichiometry_matrix}
    S = e # a hack, as sympy can only print to e = [expr] form
    {propensities}
    a = e
    return [S, a, nreactions, nvariables, ymat, Mumat, c]
      """.format(number_of_reactions=model.number_of_reactions,
                 number_of_constants=model.number_of_constants,
                 number_of_species=model.number_of_variables,
                 stoichiometry_matrix=sympy.python(model.stoichiometry_matrix).replace('\n', '\n    '),
                 propensities=sympy.python(model.propensities).replace('\n', '\n    '))

    return model_str

def parse_model(input_filename, output_file):

    infile = open(input_filename)
    try:
        D = infile.readlines()    #read input data
    finally:
        infile.close()


    stoichiometry_matrix = None
    propensities = None

    # Extract required information
    for i in range(0,len(D)):
        if REGEXP_NREACTIONS.match(D[i]):
            number_of_reactions = int(D[i+1].rstrip())
        if REGEXP_NCONSTANTS.match(D[i]):
            number_of_constants = int(D[i+1].rstrip())
        if REGEXP_NVARIABLES.match(D[i]):
            number_of_species = int(D[i+1].rstrip())

        if REGEXP_STOICHIOMETRY.match(D[i]):
            stoichiometry_components = D[i+1:i+1+number_of_species]
            stoichiometry_matrix = sympy.Matrix(stoichiometry_components)

            # TODO: remove this
            S = ''
            for j in range(i+1, i+1+int(number_of_species)):

                if REGEXP_S_ENTRY.match(D[j]):
                    S += str(REGEXP_S_ENTRY.match(D[j]).group(1)) +','
            S = 'Matrix('+str(number_of_species)+','+str(number_of_reactions)+',['+S.rstrip(',')+'])'

        if REGEXP_PROPENSITIES.match(D[i]):
            propensity_components = D[i+1:i+1+number_of_reactions]
            propensities = sympy.Matrix(map(index_to_symbol, propensity_components))

            a = ''
            index = 0
            for k in range(i+1, i+1+int(number_of_reactions)):
                a += '\ta['+str(index)+'] = '+D[k].rstrip() + '\n'
                index+=1
            a = str.replace(a,'y', 'ymat')

    constants = sympy.symbols(['c_{0}'.format(number_of_constants)])
    variables = sympy.symbols(['y_{0}',format(number_of_species)])
    model = Model(constants, variables, propensities, stoichiometry_matrix)


    output = open(output_file,'w')
    try:
        output.write(format_model_to_legacy_model_py(model))
        #'from sympy import Matrix\nfrom initialize_parameters import initialize_parameters\n\ndef model():\n\tnreactions = '+str(number_of_reactions)+'\n\tnrateconstants = '+str(number_of_constants)+'\n\tnvariables = '+str(number_of_species)+'\n\t[ymat, Mumat, c]=initialize_parameters(nrateconstants,nvariables)\n\tS = '+S+'\n\ta = Matrix(nreactions, 1, lambda i,j:0)\n'+a+'\n\treturn [S, a, nreactions, nvariables, ymat, Mumat, c]')
    finally:
        output.close()

if __name__ == '__main__':
    parse_model(sys.argv[1], OUTPUT_FILE)