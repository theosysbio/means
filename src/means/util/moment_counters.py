from __future__ import absolute_import, print_function
"""
MEANS Helpers
-----

This part of the package provides a function to generate all mixed and "pure" raw and
:class:`~means.core.descriptor.Moment`s up to a maximal_order.
"""

import itertools
import sympy as sp
from means.core import Moment

def generate_n_and_k_counters(max_order, species, central_symbols_prefix="M_", raw_symbols_prefix="x_"):
        r"""
        Makes a counter for central moments (n_counter) and a counter for raw moment (k_counter).
        Each is a list of :class:`~means.approximation.ode_problem.Moment`s.
        Therefore, each :class:`~means.approximation.ode_problem.Moments` is represented by both
        a vector of integer and a symbol.

        :param max_order: the maximal order of moment to be computer (will generate a list of moments up to `max_order + 1`)
        :param species: the name of the species
        :return: a pair of lists of :class:`~means.core.descriptors.Moment`s corresponding to central,
        and raw moments, respectively.
        :rtype: (list[:class:`~mea ns.core.descriptors.Moment`],list[:class:`~mea ns.core.descriptors.Moment`])
        """
        n_moments = max_order + 1
        # first order moments are always 1
        k_counter = [Moment([0] * len(species), sp.Integer(1))]
        n_counter = [Moment([0] * len(species), sp.Integer(1))]

        # build descriptors for first order raw moments aka expectations (e.g. [1, 0, 0], [0, 1, 0] and [0, 0, 1])
        descriptors = []
        for i in range(len(species)):
            row = [0]*len(species)
            row[i] = 1
            descriptors.append(row)

        # We use species name as symbols for first order raw moment
        k_counter += [Moment(d, s) for d,s in zip(descriptors, species)]

        # Higher order raw moment descriptors
        k_counter_descriptors = [i for i in itertools.product(range(n_moments + 1), repeat=len(species))
                                 if 1 < sum(i) <= n_moments]

        #this mimics the order in the original code
        k_counter_descriptors = sorted(k_counter_descriptors, key=sum)
        #k_counter_descriptors = [[r for r in reversed(k)] for k in k_counter_descriptors]

        k_counter_symbols = [sp.Symbol(raw_symbols_prefix + "_".join([str(s) for s in count]))
                             for count in k_counter_descriptors]
        k_counter += [Moment(d, s) for d,s in zip(k_counter_descriptors, k_counter_symbols)]

        #  central moments
        n_counter_descriptors = [m for m in k_counter_descriptors if sum(m) > 1]
        # arbitrary symbols
        n_counter_symbols = [sp.Symbol(central_symbols_prefix + "_".join([str(s) for s in count]))
                             for count in n_counter_descriptors]

        n_counter += [Moment(c, s) for c,s in zip(n_counter_descriptors, n_counter_symbols)]

        return n_counter, k_counter
