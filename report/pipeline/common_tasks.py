import os
import luigi
import means
import means.examples
from config import DATA_DIR
from target import PickleSerialiser, PickleSerialiserWithAdditionalParameters
from datetime import datetime

class MeansTask(luigi.Task):
    """
    A wrapper around luigi task that would automatically set the output variable to a standard used in MEANS pipelines
    """

    @property
    def _filename(self):
        # Default filename that just lists class name and parameters in dashes
        class_ = self.__class__.__name__
        params = self.get_params()
        param_values = [getattr(self, x[0]) for x in params]
        params_str = '-'.join(map(str, param_values))
        return '{0}-{1}.pickle'.format(class_, params_str)

    def _return_object(self):
        raise NotImplementedError

    def output(self):
        # Enforce all data to be in data dir
        return PickleSerialiserWithAdditionalParameters(os.path.join(DATA_DIR, self._filename))

    def run(self):
        # Poor man's timing
        start = datetime.now()
        answer = self._return_object()
        end = datetime.now()
        runtime = (end-start).total_seconds()

        # Store both the object and runtime
        self.output().dump(answer, runtime=runtime)



class Model(MeansTask):
    """
    Creating this as a task, in case one needs to generate them
    """

    name = luigi.Parameter()
    _SUPPORTED_MODELS = {'p53': means.examples.MODEL_P53,
                         'hes1': means.examples.MODEL_HES1}

    def _return_object(self):
        return self._SUPPORTED_MODELS[self.name]

class MEAProblem(MeansTask):

    model_name = luigi.Parameter()
    max_order = luigi.IntParameter()
    closure = luigi.Parameter()
    multivariate = luigi.BooleanParameter(default=True)

    def requires(self):
        return Model(self.model_name)

    def _return_object(self):
        model = self.input().load()

        # Scalar closure currently does not support univariate/multivariate
        if self.closure != 'scalar':
            kwargs = {'multivariate': self.multivariate}
        else:
            kwargs = {}
        problem = means.mea_approximation(model, self.max_order, closure=self.closure,
                                          **kwargs)

        return problem


class Simulation(MeansTask):
    pass

if __name__ == '__main__':
    luigi.run()