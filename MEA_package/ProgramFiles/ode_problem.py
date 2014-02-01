
import sympy
import numpy as np
from sympyhelpers import to_list_of_symbols, to_sympy_column_matrix


class ODEProblem(object):
    """
    Stores the left and right hand side equations to be simulated

    """

    # These are private (as indicated by __, the code is a bit messier, but we can ensure immutability this way)
    __right_hand_side = None
    __right_hand_side_as_function = None  # Buffer to cache rhs as function
    __left_hand_side = None
    __moment_dic = None
    __constants = None
    __ordered_moments = None

    def __init__(self, method, left_hand_side, right_hand_side, constants, moments):
        """
        Creates a `ODEProblem` object that stores the problem to be simulated/used for inference
        :param method: a string describing the method used to generate the problem.
        Currently, 'MEA' and 'LNA' are supported"
        :param left_hand_side: the left hand side of equations
        :param right_hand_side: the right hand side of equations
        :param constants: the constants of the model
        :param moments: the moments as a list of n-tuple, where n is the number of species
        """
        self.__left_hand_side = to_sympy_column_matrix(left_hand_side)
        self.__right_hand_side = to_sympy_column_matrix(right_hand_side)
        self.__constants = to_list_of_symbols(constants)
        self.__moment_dic = self.make_moment_dic(moments)
        self.__ordered_moments = moments
        self.__method = method

        self.validate()
#
    def make_moment_dic(self, moments):
        dic_out = dict()
        for i,m in enumerate(moments):
            dic_out[m] = i
        return dic_out


    def validate(self):
        """
        Validates whether the particular model is created properly
        """
        if self.left_hand_side.rows != self.right_hand_side.rows:
            raise ValueError("There are {0} left hand side equations and {1} right hand side equations. "
                             "The same number is expected.".format(self.left_hand_side.rows, self.right_hand_side.rows))
        if self.__method != "MEA" and self.__method != "LNA":
            raise ValueError("Only MEA or LNA methods are supported. The method '{0}' is unknown".format(self.__method))

        if self.__method == "MEA":
            if self.left_hand_side.rows != len(self.__moment_dic):
                 raise ValueError("There are {0} equations and {1} moments. "
                                  "For MEA problems, the same number is expected.".format(self.left_hand_side.rows, len(self.__moment_dic)))


    # Expose public interface for the specified instance variables
    # Note that all properties here are "getters" only, thus assignment won't work
    @property
    def left_hand_side(self):
        return self.__left_hand_side

    @property
    def variables(self):
        return to_list_of_symbols(self.__left_hand_side)

    @property
    def number_of_species(self):
        # TODO: there must be a better way to do this i.e. without counting in a loop
        # (this is how it was done in legacy way)
        return sum([str(x).startswith('y_') for x in self.left_hand_side])

    @property
    def right_hand_side(self):
        return self.__right_hand_side

    @property
    def constants(self):
        return self.__constants

    @property
    def method(self):
        return self.__method

    @property
    def moment_dic(self):
        return self.__moment_dic

    @property
    def ordered_moments(self):
        # TODO: consider removing this
        return self.__ordered_moments

    @property
    def rhs_as_function(self):
        if self.__right_hand_side_as_function is None:
            self.__right_hand_side_as_function = sympy.lambdify(self.constants + self.variables,
                                                                self.right_hand_side)

        return self.__right_hand_side_as_function

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

    method = lines[0].rstrip()

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
        right_hand_side = sympy.Matrix([sympy.sympify(l) for l in all_fields[STRING_RIGHT_HAND]])
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
        moments = [tuple(eval(l)) for l in all_fields[STRING_MOM]]
    except KeyError:
        print 'The field "' + STRING_CONSTANT + '" is not in the input file "' + input_filename +'"'
        raise

    return ODEProblem(method, left_hand_side, right_hand_side, constants, moments)


class ODEProblem_writer(object):
    def __init__(self, problem):
        self.__problem = problem
        self.__STRING_RIGHT_HAND = 'RHS of equations:'
        self.__STRING_LEFT_HAND = 'LHS:'
        self.__STRING_CONSTANT = 'Constants:'
        self.__N_VARIABLE = 'Number of variables:'
        self.__N_MOMENTS = 'Number of moments:'
        self.__N_EQS = 'Number of equations:'
        self.__STRING_MOM = 'List of moments:'
        self.__TIME_TAKEN = 'Time taken (s):'

    def write_to(self, output_file):
        lines = [self.__problem.method]

        lines += [self.__STRING_RIGHT_HAND]
        lines += [str(expr) for expr in self.__problem.right_hand_side]

        lines += [self.__STRING_LEFT_HAND]
        lines += [str(expr) for expr in self.__problem.left_hand_side]

        lines += [self.__STRING_CONSTANT]
        lines += [str(expr) for expr in self.__problem.constants]

        # get info from moments
        sum_moms = [sum(m) for m in self.__problem.moment_dic.keys()]
        n_var = len([s for s in sum_moms if s == 1])
        n_mom = max(sum_moms)


        lines += [self.__N_VARIABLE, str(n_var)]
        lines += [self.__N_MOMENTS, str(n_mom)]
        lines += [self.__TIME_TAKEN + "  TODO"]
        lines += [self.__N_EQS, str(self.__problem.left_hand_side)]




        lines += [self.__STRING_MOM]
        lines += [str(m) for m in self.__problem.moment_dic.keys()]

        with open(output_file, 'w') as file:
            for l in lines:
                file.write(l+"\n")

