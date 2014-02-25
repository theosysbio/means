import re
import sympy
from means.io.serialise import SerialisableObject

from means.util.sympyhelpers import to_sympy_matrix, to_sympy_column_matrix, to_list_of_symbols, sympy_expressions_equal

__all__ = ['Model']

class Model(SerialisableObject):
    """
    Stores the model of reactions we want to analyse
    """

    # These are private (as indicated by __, the code is a bit messier, but we can ensure immutability this way)
    __constants = None
    __species = None
    __propensities = None
    __stoichiometry_matrix = None

    yaml_tag = u'!model'

    def __init__(self, constants, species, propensities, stoichiometry_matrix):
        """
        Creates a `Model` object that stores the model of reactions we want to analyse
        :param constants: constants of the model, as `sympy` symbols
        :param species: variables of the model, as `sympy` symbols, i.e. speces
        :param propensities: a matrix of propensities for each of the reaction in the model.
        :param stoichiometry_matrix: stoichiometry matrix for the model
        """
        self.__constants = to_list_of_symbols(constants)
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
                             'S ({0.rows}x{0.cols}): {0!r} , propensities ({1.rows}x{1.cols}): {1!r}'.format(self.stoichiometry_matrix, self.propensities))

        if self.stoichiometry_matrix.rows != len(self.species):
            raise ValueError('There must be a row in stoichiometry matrix for each variable. '
                             'S ({0.rows}x{0.cols}): {0!r}, variables: {1!r}'.format(self.stoichiometry_matrix, self.species))

    # Expose public interface for the specified instance variables
    # Note that all properties here are "getters" only, thus assignment won't work
    @property
    def constants(self):
        return self.__constants

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
    def number_of_constants(self):
        return len(self.__constants)


    def __unicode__(self):
        return u"{0.__class__!r}\n" \
               u"Species: {0.species!r}\n" \
               u"Constants: {0.constants!r}\n" \
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
        lines.append(r"\text{{Constants}} &= {0} \\".format(sympy.latex(self.constants)))
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
        mapping = [('species', map(str, data.species)), ('constants', map(str, data.constants)),
                   ('stoichiometry_matrix', map(lambda x: map(int, x), data.stoichiometry_matrix.tolist())),
                   ('propensities', map(str, data.propensities))]

        return dumper.represent_mapping(cls.yaml_tag, mapping)