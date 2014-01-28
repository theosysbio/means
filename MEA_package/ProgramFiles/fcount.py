import itertools
def fcount(n_moments,n_vars):

    """
    :param n_moments: the maximal order of moment to be computer
    :param n_vars: the number of variables
    :return: a pair of tuples. the first element contains the all the permutations,
    whilst the second element does not have the first order (e.g. {0,0,1})
    """

    #todo we should really make a list of tuples here, but we kept list of list otherwise test fail...
    m_counter = [list(i) for i in itertools.product(range(0, n_moments + 1), repeat = n_vars) if sum(i) <= n_moments]
    counter = [i for i in m_counter if sum(i) != 1]
    return (counter, m_counter)
