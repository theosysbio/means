from time import time


class ApproximationBaseClass(object):
    """
    A base class providing a framework to generate an :class:`~means.approximation.ode_problem.ODEProblem`
    from an explicit :class:`~means.model.model.Model`
    (and, possibly, parameters)
    """
    def __init__(self, model):
        """
        Initialise the approximator.

        :param model: Model to approximate
        :type model: :class:`~means.model.Model`
        """
        self.__model = model
        self.time_last_run = None

    @property
    def model(self):
        """
        The model that is used in approximation
        """
        return self.__model

    def _wrapped_run(self):
        raise NotImplementedError

    def run(self):
        """
        Perform the approximation. Return a constructed :class:`~means.approximation.ode_problem.ODEProblem` object.

        :return: a constructed set of equations, encoded in :class:`~means.approximation.ode_problem.ODEProblem` object.
        :rtype: :class:`~means.approximation.ode_problem.ODEProblem`
        """
        t0 = time()
        out_problem = self._wrapped_run()
        self.time_last_run = time() - t0
        return out_problem