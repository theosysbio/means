"""
Model
-----

`Model` objects describe a system in terms of **stochastic** reaction propensities/rates, species/variables,
constants and a stoichiometry matrix.
Generally, describing a model is a pre-requisite for any subsequent analysis.

An example showing the p53 model could be encoded:

>>> from means import Model
>>> my_model = Model(parameters=['c_0',   # P53 production rate
>>>                             'c_1',   # MDM2-independent p53 degradation rate
>>>                             'c_2',   # saturating p53 degradation rate
>>>                             'c_3',   # P53-dependent MDM2 production rate
>>>                             'c_4',   # MDM2 maturation rate
>>>                             'c_5',   # MDM2 degradation rate
>>>                             'c_6'],  # P53 threshold of degradation by MDM2
>>>                  species=['y_0',   # Concentration of p53
>>>                           'y_1',   # Concentration of MDM2 precursor
>>>                           'y_2'],  # Concentration of MDM2
>>>                  stoichiometry_matrix=[[1, -1, -1, 0, 0, 0],
>>>                                        [0, 0, 0, 1, -1, 0],
>>>                                        [0, 0, 0, 0, 1, -1]],
>>>                  propensities=['c_0',
>>>                                'c_1*y_0',
>>>                                'c_2*y_2*y_0/(y_0+c_6)',
>>>                                'c_3*y_0',
>>>                                'c_4*y_1',
>>>                                'c_5*y_2'])

Printing the model to ensure everything is all right:

>>> print my_model

Typically, a model would be used for approximation (e.g.
:mod:`~means.approximation.mea.moment_expansion_approximation`, or
:mod:`~means.approximation.lna.lna`)
and stochastic simulations (e.g. :mod:`~means.simulation.ssa`).

-------------
"""

import sympy
from means.io.latex import LatexPrintableObject
from means.io.serialise import SerialisableObject
from means.util.sympyhelpers import to_sympy_matrix, to_sympy_column_matrix, to_list_of_symbols, sympy_expressions_equal


class Model(SerialisableObject, LatexPrintableObject):
    """
    Stores the model of reactions we want to analyse
    """

    # These are private (as indicated by __, the code is a bit messier, but we can ensure immutability this way)
    __parameters = None
    __species = None
    __propensities = None
    __stoichiometry_matrix = None

    yaml_tag = u'!model'

    def __init__(self, species, parameters, propensities, stoichiometry_matrix):
        r"""
        Creates a `Model` object that stores the model of reactions we want to analyse
        :param species: variables of the model, as `sympy.Symbol`s, i.e. species
        :param parameters: parameters of the model, as `sympy` symbols
        :param propensities: a matrix of propensities for each of the reaction in the model.
        :param stoichiometry_matrix: stoichiometry matrix for the model
        """
        self.__parameters = to_list_of_symbols(parameters)
        self.__species = to_list_of_symbols(species)
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
                             'S ({0.rows}x{0.cols}): {0!r} , '
                             'propensities ({1.rows}x{1.cols}): {1!r}'.format(self.stoichiometry_matrix,
                                                                              self.propensities))

        if self.stoichiometry_matrix.rows != len(self.species):
            raise ValueError('There must be a row in stoichiometry matrix for each variable. '
                             'S ({0.rows}x{0.cols}): {0!r}, variables: {1!r}'.format(self.stoichiometry_matrix,
                                                                                     self.species))

        seen_free_symbols = set()
        parameters = set(self.parameters)
        species = set(self.species)

        # Check if there are any parameters in both lists
        intersection = parameters & species
        if intersection:
            raise ValueError("Some symbols are in both parameters and species lists")

        both = parameters | species
        for row in self.propensities:
            free_symbols = row.free_symbols
            # Do not check the seen symbols twice
            free_symbols = free_symbols - seen_free_symbols
            for symbol in free_symbols:
                if symbol not in both:
                    raise ValueError('Propensity {0!r} '
                                     'contains a free symbol {1!r} '
                                     'that is not in listed in parameters or species lists '
                                     'Parameters: {2!r}; '
                                     'Species: {3!r}'.format(row, symbol,
                                                             self.parameters,
                                                             self.species))

            seen_free_symbols.update(free_symbols)


    # Expose public interface for the specified instance variables
    # Note that all properties here are "getters" only, thus assignment won't work
    @property
    def parameters(self):
        return self.__parameters

    @property
    def species(self):
        return self.__species

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
    def number_of_species(self):
        return len(self.__species)

    @property
    def number_of_parameters(self):
        return len(self.__parameters)


    def __unicode__(self):
        return u"{0.__class__!r}\n" \
               u"Species: {0.species!r}\n" \
               u"Parameters: {0.parameters!r}\n" \
               u"\n" \
               u"Stoichiometry matrix:\n" \
               u"{0.stoichiometry_matrix!r}\n" \
               u"\n" \
               u"Propensities:\n" \
               u"{0.propensities!r}".format(self)

    def __str__(self):
        return unicode(self).encode("utf8")

    def __repr__(self):
        return str(self)

    def _repr_latex_(self):

        lines = []
        lines.append(r"\begin{align*}")
        lines.append(r"\text{{Species}} &= {0} \\".format(sympy.latex(self.species)))
        lines.append(r"\text{{Parameters}} &= {0} \\".format(sympy.latex(self.parameters)))
        lines.append(r"\text{{Stoichiometry matrix}} &= {0} \\".format(sympy.latex(self.stoichiometry_matrix)))
        lines.append(r"\text{{Propensities}} &= {0} \\".format(sympy.latex(self.propensities)))
        lines.append(r"\end{align*}")
        return "\n".join(lines)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.species == other.species and self.propensities == other.propensities \
                   and self.stoichiometry_matrix == other.stoichiometry_matrix \
                   and sympy_expressions_equal(self.propensities,other.propensities)


    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = [('species', map(str, data.species)), ('parameters', map(str, data.parameters)),
                   ('stoichiometry_matrix', map(lambda x: map(int, x), data.stoichiometry_matrix.tolist())),
                   ('propensities', map(str, data.propensities))]

        return dumper.represent_mapping(cls.yaml_tag, mapping)


    def __hash__(self):
        hashable_payload = (tuple(self.species),
                            tuple(self.parameters),
                            tuple(self.propensities),
                            tuple(self.stoichiometry_matrix))
        return hash(hashable_payload)