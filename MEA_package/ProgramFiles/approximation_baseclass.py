import model
from time import time

class ApproximationBaseClass(object):
    def __init__(self,model):
        self.__model = model
        self.time_last_run = None

    @property
    def model(self):
        return self.__model

    #@abstractmethod
    #def __run(self, parameters):
        #raise NotImplementedError

    def run_with_params(self, parameters):
        t0 = time()
        out = self._run(parameters)
        self.time_last_run = time() - t0
        return out