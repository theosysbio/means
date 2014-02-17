import sympy
import numpy as np
from sympy.utilities.autowrap import autowrap

from means.util.sympyhelpers import to_list_of_symbols, to_sympy_column_matrix, to_sympy_matrix, to_one_dim_array
from means.util.decorators import memoised_property

class Descriptor(object):
    pass

class ODETermBase(Descriptor):
    """
    Base class for explaining terms in the ODE expressions.
    Instances of this class allow providing a description for each of the equations in the generated ODE system.
    """

    _symbol = None

    def __init__(self, symbol):
        super(ODETermBase, self).__init__()
        self._symbol = symbol

    @property
    def symbol(self):
        return self._symbol

    @property
    def descriptor(self):
        """
        Returns an uniquely identifying descriptor for this particular ODE term.
        """
        return None

    def __repr__(self):
        return str(self)

    def __str__(self):
        return unicode(self).encode('utf8')

    def __unicode__(self):
        return u'{0}({1})'.format(self.__class__.__name__, self.symbol)

    def __mathtext__(self):
        # Double {{ and }} in multiple places as to escape the curly braces in \frac{} from .format
        return r'${0}$'.format(self.symbol)


class VarianceTerm(ODETermBase):
    """
    Signifies that a particular equation generated from the model is part of a Variance Term
    """
    _position = None

    def __init__(self, symbol, position):
        """
        Creates a Descriptor for a particular ODE in the system that signifies that that particular equation
        computes the position-th term of a covariance matrix, where position is some tuple (row,column).

        It is used in LNA approximation as there we need to deal with moment and variance terms differently

        :param symbol: symbol assigned to the term
        :param position: position in the covariance matrix

        """
        super(VarianceTerm, self).__init__(symbol=symbol)
        self._position = position

    @property
    def position(self):
        return self._position

    def __unicode__(self):
        return u'{0}({1}, {2})'.format(self.__class__.__name__, self.symbol, self.position)



class Moment(ODETermBase):
    """
    An annotator for ODE expressions that describes that a particular expression in a set of ODEs corresponds to a Moment
    of the probability distribution. The particular moment is described by :attr:`Moment.n_vector`.
    """
    __n_vector = None

    def __init__(self, n_vector, symbol=None):
        """
        Creates an ODETerm that describes that a particular ODE term is a moment defined by the `n_vector`.
        Should be a vector of ints.

        TODO: figure out what "n_vector" is in mathematics-speak and use this here
        FIXME: can symbol really be optional None?????
        FIXME: symbol should be first argument to make it consistent with ODETermBase
        :param n_vector: a vector specifying the multidimensional moment
        """
        super(Moment, self).__init__(symbol=symbol)

        self.__n_vector = np.array(n_vector, dtype=int)
        self.__order = sum(self.n_vector)
        self.__descriptor = self.n_vector

    @property
    def descriptor(self):
        return self.__n_vector

    @property
    def n_vector(self):
        """
        The n_vector this moment represents
        """
        return self.__n_vector

    @property
    def order(self):
        """
        The order of the moment
        """
        return self.__order

    @property
    def is_mixed(self):
        """
        Returns whether the moment is a mixed moment, i.e. has a non-zero power to more than one species,
        or a raw moment (non-zero power to only one species).
        """
        # If moment is not mixed, it will be of form [0, ... , k, ..., 0] where k is the max order
        return self.order not in self.n_vector

    def __str__(self):
        return ', '.join(map(str, self.n_vector))

    def __hash__(self):
        # Allows moment objects to be stored as keys to dictionaries
        return hash(repr(self.n_vector))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        elif (self.n_vector != other.n_vector).any():
            return False
        elif self.symbol != self.symbol:
            return False
        else:
            return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __ge__(self, other):
        """
        A moment is said greater or equal than another iff all the values of n_vec are greater or equal.
        Mathematically: ::math::`n_i^a \ge n_i^b ~ \textrm{for all i}`
        """
        return (self.n_vector >= other.n_vector).all()
        #return all([a >= b for a, b in zip])

    def __repr__(self):
        return '{0}({1!r}, symbol={2!r})'.format(self.__class__.__name__, self.n_vector, self.symbol)



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

    def __init__(self, method, ode_lhs_terms, right_hand_side, constants):
        """
        Creates a `ODEProblem` object that stores the problem to be simulated/used for inference
        :param method: a string describing the method used to generate the problem.
        Currently, 'MEA' and 'LNA' are supported"
        :param ode_lhs_terms: the left hand side of equations as a list of `ODETerms` (e.g. `Moments`)
        :param right_hand_side: the right hand side of equations
        :param constants: the constants of the model
        """

        self.__ode_lhs_terms = ode_lhs_terms
        self.__left_hand_side = to_sympy_column_matrix(to_sympy_matrix([plhs.symbol for plhs in ode_lhs_terms]))
        self.__right_hand_side = to_sympy_column_matrix(right_hand_side)
        self.__constants = to_list_of_symbols(constants)
        self.__method = method
        self.__initialise_descriptions(ode_lhs_terms)

    #todo
    # def __eq__(self, other):
    #    return True

    def __initialise_descriptions(self, ode_lhs_terms):
        """
        Populate self.__descriptions_dict
        and self._ordered_descriptions_of_lhs_terms
        :param ode_lhs_terms:
        :return:
        """
        descriptions_dict = dict([(ode_term.symbol, ode_term) for ode_term in ode_lhs_terms])
        self.__ordered_descriptions_of_lhs_terms = ode_lhs_terms
        self.__descriptions_dict = descriptions_dict

    def validate(self):
        """
        Validates whether the ODE equations provided make sense  i.e. the number of right-hand side equations
        match the number of left-hand side equations.
        """
        if self.left_hand_side.rows != self.right_hand_side.rows:
            raise ValueError("There are {0} left hand side equations and {1} right hand side equations. "
                             "The same number is expected.".format(self.left_hand_side.rows, self.right_hand_side.rows))

    # Expose public interface for the specified instance variables
    # Note that all properties here are "getters" only, thus assignment won't work
    @property
    def ode_lhs_terms(self):
        return self.__ode_lhs_terms

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
    def method(self):
        return self.__method

    @property
    def descriptions_dict(self):
        return self.__descriptions_dict

    @property
    def ordered_descriptions(self):
        # TODO: consider removing this
        return self.__ordered_descriptions_of_lhs_terms

    @property
    def number_of_equations(self):
        return len(self.left_hand_side)

    @memoised_property
    def _right_hand_side_as_numeric_functions(self):
        all_symbols = self.constants + self.variables
        wrapping_func = lambda x: autowrap(x, args=all_symbols, language='C', backend='Cython')
        return map(wrapping_func, self.right_hand_side)

    @memoised_property
    def right_hand_side_as_function(self):
        """
        Generates and returns the right hand side of the model as a callable function that takes two parameters:
        values for variables and values for constants,
        e.g. `f(values_for_variables=[1,2,3], values_for_constants=[3,4,5])

        This function is directly used in `means.simulation.Simulation`
        :return:
        :rtype: function
        """
        wrapped_functions = self._right_hand_side_as_numeric_functions

        def f(values_for_variables, values_for_constants):
            all_values = np.concatenate((values_for_constants, values_for_variables))
            ans = np.array([w_f(*all_values) for w_f in wrapped_functions])
            return ans

        return f

    def __unicode__(self):
        equations_pretty_str = '\n\n'.join(['{0!r}:\n    {1!r}'.format(x, y) for x, y in zip(self.ode_lhs_terms,
                                                                                           self.right_hand_side)])
        return u"{0.__class__!r}\n" \
               u"Method: {0.method!r}\n" \
               u"Constants: {0.constants!r}\n" \
               u"\n" \
               u"Equations:\n\n" \
               u"{1}\n".format(self, equations_pretty_str)

    def __str__(self):
        return unicode(self).encode("utf8")

    def __repr__(self):
        return str(self)


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
        right_hand_side = sympy.Matrix(sympy.sympify([l for l in all_fields[STRING_RIGHT_HAND]]))
    except KeyError:
        print 'The field "' + STRING_RIGHT_HAND + '" is not in the input file "' + input_filename +'"'
        raise
    try:
        left_hand_side = sympy.Matrix(sympy.sympify([l for l in all_fields[STRING_LEFT_HAND]]))
    except KeyError:
        print 'The field "' + STRING_LEFT_HAND + '" is not in the input file "' + input_filename +'"'
        raise
    try:
        constants = all_fields[STRING_CONSTANT]
    except KeyError:
        print 'The field "' + STRING_CONSTANT + '" is not in the input file "' + input_filename +'"'
        raise
    try:
        n_vecs = [list(eval(l)) for l in all_fields[STRING_MOM]]
    except KeyError:
        print 'The field "' + STRING_CONSTANT + '" is not in the input file "' + input_filename +'"'
        raise

    moment_terms = [Moment(nv,lhs) for (nv,lhs) in zip(n_vecs, left_hand_side)]
    # TODO: remove this hack below, where I read the variance position from the symbol name
    # Replace this with explicitly storing them alongside list of moments
    variance_terms = []
    for lhs in left_hand_side[len(moment_terms):]:
        str_lhs = str(lhs)
        position = int(str_lhs[2]), int(str_lhs[3])
        variance_terms.append(VarianceTerm(lhs, position))
    ode_terms = moment_terms + variance_terms

    return ODEProblem(method, ode_terms, right_hand_side, constants)


class ODEProblemWriter(object):
    """
    A file writer for :class:`~means.approximation.ode_problem.ODEProblem` objects.
    """

    def __init__(self, problem):
        """
        :param problem: the problem to be written
        :type problem: :class:`~means.approximation.ode_problem.ODEProblem`
        """
        self._problem = problem
        self._STRING_RIGHT_HAND = 'RHS of equations:'
        self._STRING_LEFT_HAND = 'LHS:'
        self._STRING_CONSTANT = 'Constants:'
        self._N_VARIABLE = 'Number of variables:'
        self._N_MOMENTS = 'Number of moments:'
        self._N_EQS = 'Number of equations:'
        self._STRING_MOM = 'List of moments:'

    def build_out_string_list(self):
        """
        Makes a list of strings, one for each line, to be writen to a file later.
        :return: the list of string to be written
        """

        #empty lines are added in order to mimic the output from the original code

        left_hand_side = self._problem.ode_lhs_terms

        lines = [self._problem.method]
        lines += [""]
        lines += [self._STRING_RIGHT_HAND]
        lines += [str(expr) for expr in self._problem.right_hand_side]

        lines += [""]

        lines += [self._STRING_LEFT_HAND]
        lines += [str(lhs.symbol) for lhs in left_hand_side]

        lines += [""]

        lines += [self._STRING_CONSTANT]
        lines += [str(expr) for expr in self._problem.constants]

        n_var = self._problem.number_of_species

        lines += [self._N_VARIABLE, str(n_var)]

        # number of mom only relevant for MEA
        if(self._problem.method == "MEA"):
            n_mom = max([lhs.order for lhs in left_hand_side])
            lines += [self._N_MOMENTS, str(n_mom)]

        lines += [""]
        lines += [self._N_EQS, str(self._problem.number_of_equations)]
        lines += [""]
        lines += [self._STRING_MOM]
        lines += ["[" + str(lhs) + "]" for lhs in left_hand_side if isinstance(lhs, Moment)]
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
        left_hand_side = self._problem.ode_lhs_terms
        preamble = ["\documentclass{article}"]
        preamble += ["\usepackage[landscape, margin=0.5in, a3paper]{geometry}"]
        lines = ["\\begin{document}"]
        lines += ["\section*{%s}" % self._STRING_RIGHT_HAND]

        lines += ["$\dot {0} = {1} {2}$".format(str(sympy.latex(lhs.symbol)), str(sympy.latex(rhs)), r"\\")
                    for (rhs, lhs) in zip(self._problem.right_hand_side, left_hand_side)]

        lines += [r"\\"] * 5

        lines += ["\section*{%s}" % self._STRING_MOM]
        #ordered_moments = sorted([(i,m) for (m,i) in self._problem.moment_dic.items()])


        lines += ["$\dot {0}$: {1} {2}".format(str(sympy.latex(lhs.symbol)), str(lhs), r"\\")
                       for lhs in left_hand_side if isinstance(lhs, Moment)]

        lines += ["\end{document}"]

        return preamble + lines
