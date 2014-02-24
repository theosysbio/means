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
    parameters = sympy.symbols([p.getId() for p in libsbml_reaction.getKineticLaw().getListOfParameters()])
    return _Reaction(id_, reactants, products, kinetic_law, parameters)


def read_sbml(filename):
    import libsbml

    reader = libsbml.SBMLReader()
    document = reader.readSBML(filename)

    sbml_model = document.getModel()

    species = sympy.symbols([s.getId() for s in sbml_model.getListOfSpecies()])
    reactions = map(_parse_reaction, sbml_model.getListOfReactions())

    parameters = set()

    stoichiometry_matrix = np.zeros((len(species), len(reactions)), dtype=int)
    propensities = []
    for reaction_index, reaction in enumerate(reactions):
        parameters.update(reaction.parameters)
        reactants = reaction.reactants
        products = reaction.products
        propensities.append(reaction.propensity)

        for species_index, species_id in enumerate(species):
            net_stoichiometry = products.get(species_id, 0) - reactants.get(species_id, 0)
            stoichiometry_matrix[species_index, reaction_index] = net_stoichiometry

    # sympy does not allow sorting its parameter lists by default,
    # explicitly tell to sort by str representation
    sorted_parameters = sorted(parameters, key=str)
    model = Model(sorted_parameters, species, propensities, stoichiometry_matrix)

    return model