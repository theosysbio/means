from collections import namedtuple
import os
import sympy
import numpy as np
from means.core.model import Model

_Reaction = namedtuple('_REACTION', ['id', 'reactants', 'products', 'propensity', 'parameters'])

def _sbml_like_piecewise(*args):

    if len(args) % 2 == 1:
        # Add a final True element you can skip in SBML
        args += (True,)

    sympy_args = []

    for i in range(len(args)/2):
        # We need to group args into tuples of form
        # (value, condition)
        # SBML usually outputs them in form (value, condition, value, condition, value ...)
        sympy_args.append((args[i*2], args[i*2+1]))

    return sympy.Piecewise(*sympy_args)

def _sympify_kinetic_law_formula(formula):

    # We need to define some namespace hints for sympy to deal with certain functions in SBML formulae
    # For instance, `eq` in formula should map to `sympy.Eq`

    namespace = {'eq': sympy.Eq,
                 'neq': sympy.Ne,
                 'floor': sympy.floor,
                 'ceiling': sympy.ceiling,
                 'gt': sympy.Gt,
                 'lt': sympy.Lt,
                 'geq': sympy.Ge,
                 'leq': sympy.Le,
                 'pow': sympy.Pow,
                 'piecewise': _sbml_like_piecewise}

    return sympy.sympify(formula, locals=namespace)

def _parse_reaction(libsbml_reaction):
    id_ = libsbml_reaction.getId()
    reactants = {sympy.Symbol(r.getSpecies()): r.getStoichiometry() for r in libsbml_reaction.getListOfReactants()}
    products = {sympy.Symbol(p.getSpecies()): p.getStoichiometry() for p in libsbml_reaction.getListOfProducts()}
    kinetic_law = _sympify_kinetic_law_formula(libsbml_reaction.getKineticLaw().getFormula())
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

    if not os.path.exists(filename):
        raise IOError('File {0!r} does not exist'.format(filename))

    reader = libsbml.SBMLReader()
    document = reader.readSBML(filename)

    sbml_model = document.getModel()
    if not sbml_model:
        raise ValueError('Cannot parse SBML model from {0!r}'.format(filename))

    species = sympy.symbols([s.getId() for s in sbml_model.getListOfSpecies()])
    compartments = sympy.symbols([s.getId() for s in sbml_model.getListOfCompartments()])
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

    # We need to concatenate compartment names and parameters as in our framework we cannot differentiate the two
    compartments_and_parameters = compartments + sorted_parameters
    model = Model(compartments_and_parameters, species, propensities, stoichiometry_matrix)

    return model