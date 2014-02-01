
import sympy
import numpy as np
from sympyhelpers import to_list_of_symbols, to_sympy_column_matrix

class ODETermBase(object):
    pass

class Moment(ODETermBase):
    __n_vector = None

    def __init__(self, n_vector):
        """
        Creates an ODETerm that describes that a particular ODE term is a moment defined by the `n_vector`.
        Should be a vector of ints.

        TODO: figure out what "n_vector" is in mathematics-speak and use this here
        :param n_vector: a vector specifying the multidimensional moment
        """
        self.__n_vector = np.array(n_vector, dtype=int)

    @property
    def n_vector(self):
        return self.__n_vector

    @property
    def order(self):
        """
        Returns the order of the moment
        """
        return sum(self.n_vector)

    def __repr__(self):
        return '{0}({1!r})'.format(self.__class__.__name__, self.n_vector)

    def __str__(self):
        return ', '.join(map(str, self.n_vector))

class ODEProblem(object):
    """
    Stores the left and right hand side equations to be simulated

    """

    # These are private (as indicated by __, the code is a bit messier, but we can ensure immutability this way)
    __right_hand_side = None
    __left_hand_side = None
    __descriptions_dict = None
    __constants = None
    __ordered_descriptions_of_lhs_terms = None

    def __init__(self, left_hand_side, right_hand_side, constants, description_of_lhs_terms=None):
        """
        Creates a `ODEProblem` object that stores the problem to be simulated/used for inference
        :param left_hand_side: the left hand side of equations
        :param right_hand_side: the right hand side of equations
        :param constants: the constants of the model
        :param description_of_lhs_terms: descriptions of the terms in the left hand side of equations.
                                         Should be a dictionary of symbol -> description pairs
        """
        self.__left_hand_side = to_sympy_column_matrix(left_hand_side)
        self.__right_hand_side = to_sympy_column_matrix(right_hand_side)
        self.__constants = to_list_of_symbols(constants)

        self.__initialise_descriptions(description_of_lhs_terms)

        self.validate()

    def __initialise_descriptions(self, description_of_lhs_terms):
        """
        Populate self.__descriptions_dict
        and self._ordered_descriptions_of_lhs_terms
        :param description_of_lhs_terms:
        :return:
        """
        # NB: getting left hand side from self, rather than passing it from above as
        # we need to make sure that left_hand_side here is a list of symbols
        left_hand_side = self.left_hand_side

        if description_of_lhs_terms:

            # Validate the description_of_lhs_terms first:
            for key in description_of_lhs_terms.keys():
                symbolic_key = sympy.Symbol(key) if isinstance(key, basestring) else key
                if symbolic_key not in left_hand_side:
                    raise KeyError('Provided description key {0!r} '
                                   'is not in LHS equations {1!r}'.format(key, left_hand_side))

            ordered_descriptions = []
            for lhs in left_hand_side:
                try:
                    lhs_description = description_of_lhs_terms[lhs]
                except KeyError:
                    lhs_description = description_of_lhs_terms.get(str(lhs), None)
                ordered_descriptions.append(lhs_description)
        else:
            ordered_descriptions = [None] * len(left_hand_side)

        self.__descriptions_dict = dict(zip(left_hand_side, ordered_descriptions))
        self.__ordered_descriptions_of_lhs_terms = ordered_descriptions

    def validate(self):
        """
        Validates whether the particular model is created properly
        """
        if self.left_hand_side.rows != self.right_hand_side.rows:
            raise ValueError("There are {0} left hand side equations and {0} right hand side equations. "
                             "The same number is expected.".format(self.left_hand_side.rows, self.right_hand_side.rows))

        # TODO: Below is true for MEA, but not true for LNA
        # if self.left_hand_side.rows != len(self.__moment_dic):
        #     raise ValueError("There are {0} equations and {1} moments. "
        #                      "The same number is expected.".format(self.left_hand_side.rows, len(self.__moment_dic)))

    # Expose public interface for the specified instance variables
    # Note that all properties here are "getters" only, thus assignment won't work
    @property
    def left_hand_side(self):
        return self.__left_hand_side

    @property
    def variables(self):
        return to_list_of_symbols(self.__left_hand_side)

    # TODO: I don't think species_* methods should be part of ODEProblem, better for it to be unaware of description meanings
    @property
    def species_terms(self):
        return filter(lambda x: isinstance(x[1], Moment) and x[1].order == 1, self.descriptions_dict.iteritems())
    @property
    def number_of_species(self):
        return len(self.species_terms)

    @property
    def right_hand_side(self):
        return self.__right_hand_side

    @property
    def constants(self):
        return self.__constants

    @property
    def descriptions_dict(self):
        return self.__descriptions_dict

    @property
    def ordered_descriptions(self):
        # TODO: consider removing this
        return self.__ordered_descriptions_of_lhs_terms

    def right_hand_side_as_function(self, values_for_constants):
        """
        Returns the right hand side of the model as a callable function with constant terms i.e. `(c_1, c_2, etc.)` set
        from values_for_constants.

        The function returned takes a vector of values for the remaining variables, e.g. `f([1,2,3])`

        :param values_for_constants:
        :return:
        """
        values_for_constants = np.array(values_for_constants)
        assert(values_for_constants.shape == (len(self.constants),))
        rhs_function = sympy.lambdify(self.constants + self.variables, self.right_hand_side)

        def f(values_for_variables):
            all_values = np.concatenate((values_for_constants, values_for_variables))
            return rhs_function(*all_values)

        return f

def parse_problem(input_filename, from_string=False):
    """
    Parses model from the `input_filename` file and returns it
    :param input_filename:
    :param from_string: True if a string containing the file data is passed instead of a filename
    :return: Parsed `ODEProblem` object
    :rtype: ODEProblem
    """

    # Strings to identify appropriate fields of the file
    STRING_RIGHT_HAND = 'RHS of equations:'
    STRING_LEFT_HAND = 'LHS:'
    STRING_CONSTANT = 'Constants:'
    STRING_MOM = 'List of moments:'

    if not from_string:
        infile = open(input_filename)
        try:
            lines = infile.readlines()    #read input data
        finally:
            infile.close()
    else:
        lines = input_filename.split("\n")

    all_fields = dict()
    field = None
   # cut the file into chunks. The lines containing ":" are field headers
    for i, line in enumerate(lines):
        if ":" in line:
            field = line.rstrip()
            all_fields[field]=[]
        elif field:
            rsline = line.rstrip()
            if rsline:
                all_fields[field].append(rsline)

    # now we query all the fields we need

    try:
        right_hand_side = sympy.Matrix([sympy.simplify(l) for l in all_fields[STRING_RIGHT_HAND]])
    except KeyError:
        print 'The field "' + STRING_RIGHT_HAND + '" is not in the input file "' + input_filename +'"'
        raise
    try:
        left_hand_side = sympy.Matrix([l for l in all_fields[STRING_LEFT_HAND]])
    except KeyError:
        print 'The field "' + STRING_LEFT_HAND + '" is not in the input file "' + input_filename +'"'
        raise
    try:
        constants = all_fields[STRING_CONSTANT]
    except KeyError:
        print 'The field "' + STRING_CONSTANT + '" is not in the input file "' + input_filename +'"'
        raise
    try:
        moments = dict(zip(left_hand_side, [Moment(list(eval(l))) for l in all_fields[STRING_MOM]]))
    except KeyError:
        print 'The field "' + STRING_CONSTANT + '" is not in the input file "' + input_filename +'"'
        raise

    return ODEProblem(left_hand_side, right_hand_side, constants, moments)

