
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
        dict_out = dict(zip(moments, range(len(moments))))
        return dict_out


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

            print self.right_hand_side

            print self.constants + self.variables


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


class ODEProblemWriter(object):
    """
    A class to write the resulting "ODEProblems" in a text file.

    """

    def __init__(self, problem, run_time="unknown"):
        """

        :param problem: an ODEProblem object to be written
        :param run_time: the time taken to formulate the problem (optional)
        """
        self._problem = problem
        self._run_time = run_time
        self._STRING_RIGHT_HAND = 'RHS of equations:'
        self._STRING_LEFT_HAND = 'LHS:'
        self._STRING_CONSTANT = 'Constants:'
        self._N_VARIABLE = 'Number of variables:'
        self._N_MOMENTS = 'Number of moments:'
        self._N_EQS = 'Number of equations:'
        self._STRING_MOM = 'List of moments:'
        self._TIME_TAKEN = 'Time taken (s):'

    def build_out_string_list(self):
        """
        Makes a list of strings, one for each line, to be writen to a file later.
        :return: the list of string to be written
        """

        #empty lines are added in order to mimic the output from the original code
        lines = [self._problem.method]

        lines += [""]

        lines += [self._STRING_RIGHT_HAND]
        lines += [str(expr) for expr in self._problem.right_hand_side]

        lines += [""]

        lines += [self._STRING_LEFT_HAND]
        lines += [str(expr) for expr in self._problem.left_hand_side]

        lines += [""]

        lines += [self._STRING_CONSTANT]
        lines += [str(expr) for expr in self._problem.constants]

        # get info from moments
        sum_moms = [sum(m) for m in self._problem.moment_dic.keys()]
        n_var = len([s for s in sum_moms if s == 1])
        n_mom = max(sum_moms)

        lines += [self._N_VARIABLE, str(n_var)]
        lines += [self._N_MOMENTS, str(n_mom)]
        lines += [self._TIME_TAKEN + "  {0}".format(self._run_time)]

        lines += [""]

        lines += [self._N_EQS, str(self._problem.left_hand_side)]

        lines += [""]


        ordered_moments = sorted([(i,m) for (m,i) in self._problem.moment_dic.items()])

        lines += [self._STRING_MOM]
        lines += [str(list(mom)) for (lhs,mom) in ordered_moments]
        return lines

    def write_to(self, output_file):

        """
        Public method to write the problem to a given file
        :param output_file: the name of the file. It will be created if needed
        """
        lines = self.build_out_string_list()
        with open(output_file, 'w') as file:
            for l in lines:
                file.write(l+"\n")

class ODEProblemLatexWriter(ODEProblemWriter):
    """
    A class to write formated LaTeX equations representing a problem
    """
    def build_out_string_list(self):


        """
        Overrides the default method and provides latex expressions instead of plain text
        :return: LaTeX formated list of strings
        """
        preamble = ["\documentclass{article}"]
        preamble += ["\usepackage[landscape, margin=0.5in, a3paper]{geometry}"]
        lines = ["\\begin{document}"]
        lines += ["\section*{%s}" % self._STRING_RIGHT_HAND]

        lines += ["$\dot {0} = {1} {2}$".format(str(sympy.latex(lhs)), str(sympy.latex(rhs)), r"\\")
                    for (rhs, lhs) in zip(self._problem.right_hand_side, self._problem.left_hand_side)]

        lines += [r"\\"] * 5

        lines += ["\section*{%s}" % self._STRING_MOM]
        ordered_moments = sorted([(i,m) for (m,i) in self._problem.moment_dic.items()])


        lines += ["$\dot {0}$: {1} {2}".format(str(sympy.latex(lhs)), str(list(mom)), r"\\")
                       for (lhs,mom) in ordered_moments]

        lines += ["\end{document}"]

        return preamble + lines
