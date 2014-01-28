import re
from sympyhelpers import to_sympy_matrix, to_sympy_column_matrix, to_list_of_symbols
import sympy



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
            raise ValueError('There must be a column in stoichiometry matrix '
                             'for each row in propensities matrix. '
                             'S ({0.rows}x{0.cols}): {0!r} , propensities ({1.rows}x{1.cols}): {1!r}'.format(self.stoichiometry_matrix, self.propensities))

        if self.stoichiometry_matrix.rows != len(self.variables):
            raise ValueError('There must be a row in stoichiometry matrix for each variable. '
                             'S ({0.rows}x{0.cols}): {0!r}, variables: {1!r}'.format(self.stoichiometry_matrix, self.variables))

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

def parse_model(input_filename):
    """
    Parses model from the `input_filename` file and returns it
    :param input_filename:
    :return: Parsed model object
    :rtype: Model
    """

    def index_to_symbol(indexed_string):
        """
        Helper function that replaces indexed form to symbols, e.g. would replace c[0] to c_0
        :param indexed_string: string containing several indexed vectors
        :return:
        """
        return re.sub("(\w+)\[(\d+)\]", r"\1_\2", indexed_string)

    # Strings to identify appropriate sections of the file
    STRING_NREACTIONS = 'Number of reactions'
    STRING_NCONSTANTS = 'Number of rate constants'
    STIRNG_NVARIABLES = 'Number of variables'
    STRING_STOICHIOMETRY = 'Stoichiometry'
    STIRNG_PROPENSITIES = 'Reaction propensities'

    infile = open(input_filename)
    try:
        lines = infile.readlines()    #read input data
    finally:
        infile.close()

    stoichiometry_matrix = None
    propensities = None

    # TODO: get rid of this nasty loop
    # Extract required information
    for i, line in enumerate(lines):
        if STRING_NREACTIONS in line:
            number_of_reactions = int(lines[i+1].rstrip())
        if STRING_NCONSTANTS in line:
            number_of_constants = int(lines[i+1].rstrip())
        if STIRNG_NVARIABLES in line:
            number_of_species = int(lines[i+1].rstrip())

        if STRING_STOICHIOMETRY in line:
            stoichiometry_components = map(lambda x: x.rstrip().strip('[]').split(','), lines[i+1:i+1+number_of_species])
            stoichiometry_matrix = sympy.Matrix(stoichiometry_components)

        if STIRNG_PROPENSITIES in line:
            propensity_components = lines[i+1:i+1+number_of_reactions]
            propensities = sympy.Matrix(map(index_to_symbol, propensity_components))

    constants = sympy.symbols(['c_{0}'.format(i) for i in xrange(number_of_constants)])
    variables = sympy.symbols(['y_{0}'.format(i) for i in xrange(number_of_species)])
    model = Model(constants, variables, propensities, stoichiometry_matrix)

    return model