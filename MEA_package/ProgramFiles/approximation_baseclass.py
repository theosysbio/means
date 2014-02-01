import model
from time import time

class ApproximationBaseClass(object):
    """
    A base class providing a framework to generate an `ODEProblem` from an explicit `Model` (and, possibly, parameters)
    """
    def __init__(self,model):
        self.__model = model
        self.time_last_run = None

    @property
    def model(self):
        return self.__model


    def _run(self, parameters):
        raise NotImplementedError

    def run_with_params(self, parameters):
        """
        This is a wrapper around `_run()` which allow to record the time taken by the last `_run()` call.
        :param parameters: some undefined parameters that the derived class could need. For example,
        for MEA, a maximal number of moment is needed.
        :return: a `ODEProblem` object.
        """
        t0 = time()
        out_problem = self._run(parameters)
        self.time_last_run = time() - t0
        return out_problem