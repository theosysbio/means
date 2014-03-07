from collections import namedtuple
import sympy
import numpy as np
from means.model import Model

_Reaction = namedtuple('_REACTION', ['id', 'reactants', 'products', 'propensity', 'parameters'])

def _parse_reaction(libsbml_reaction):
    id_ = libsbml_reaction.getId()
    reactants = {sympy.Symbol(r.getSpecies()): r.getStoichiometry() for r in libsbml_reaction.getListOfReactants()}
    products = {sympy.Symbol(p.getSpecies()): p.getStoichiometry() for p in libsbml_reaction.getListOfProducts()}
    kinetic_law = sympy.sympify(libsbml_reaction.getKineticLaw().getFormula())
    # This would only work for SBML Level 3, prior levels do not have parameters within kinetic law
    parameters = sympy.symbols([p.getId() for p in libsbml_reaction.getKineticLaw().getListOfParameters()])

    return _Reaction(id_, reactants, products, kinetic_law, parameters)


def read_sbml(filename):
    """
    Read the model from a SBML file.

    :param filename: SBML filename to read the model from
    :return: A fully created model instance
    :rtype: ::class::`means.Model`
    """
    import libsbml

    reader = libsbml.SBMLReader()
    document = reader.readSBML(filename)

    sbml_model = document.getModel()

    species = sympy.symbols([s.getId() for s in sbml_model.getListOfSpecies()])
    reactions = map(_parse_reaction, sbml_model.getListOfReactions())


    # getListOfParameters is an attribute of the model for SBML Level 1&2
    parameters = sympy.symbols([p.getId() for p in sbml_model.getListOfParameters()])

    if not parameters:
        track_local_parameters = True
        parameters = set()
    else:
        track_local_parameters = False

    stoichiometry_matrix = np.zeros((len(species), len(reactions)), dtype=int)
    propensities = []
    for reaction_index, reaction in enumerate(reactions):
        if track_local_parameters:
            parameters.update(reaction.parameters)
        reactants = reaction.reactants
        products = reaction.products
        propensities.append(reaction.propensity)
        for species_index, species_id in enumerate(species):
            net_stoichiometry = products.get(species_id, 0) - reactants.get(species_id, 0)
            stoichiometry_matrix[species_index, reaction_index] = net_stoichiometry


    if track_local_parameters:
        # sympy does not allow sorting its parameter lists by default,
        # explicitly tell to sort by str representation
        sorted_parameters = sorted(parameters, key=str)
    else:
        sorted_parameters = parameters

    model = Model(sorted_parameters, species, propensities, stoichiometry_matrix)

    return model