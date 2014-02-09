from time import time


class ApproximationBaseClass(object):
    """
    A class of explicit generators for ordinary differential equations required to simulate the model provided.
    """
    def __init__(self, model):
        """
        Initialise the approximator.

        :param model: Model to approximate
        :type model: :class:`~means.model.Model`
        """
        self.__model = model

    @property
    def model(self):
        """
        The model that is used in approximation
        """
        return self.__model

    def run(self):
        """
        Perform the approximation. Return a constructed :class:`~means.approximation.ode_problem.ODEProblem` object.

        :return: a constructed set of equations, encoded in :class:`~means.approximation.ode_problem.ODEProblem` object.
        :rtype: :class:`~means.approximation.ode_problem.ODEProblem`
        """
        raise NotImplementedError