import os
import luigi
import means
import means.examples
from config import DATA_DIR
from target import PickleSerialiser, PickleSerialiserWithAdditionalParameters
from datetime import datetime
import numpy as np
import logging
logger = logging.getLogger('luigi-interface')

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
        try:
            # Make sure we have only one instance of output object
            return self.__output
        except AttributeError:
            # Force the directory structure <DATA_DIR>/<CLASS>/<FILENAME>
            output = PickleSerialiserWithAdditionalParameters(os.path.join(DATA_DIR,
                                                              self.__class__.__name__,
                                                              self._filename))
            self.__output = output
            return output

    def run(self):
        # Poor man's timing
        start = datetime.now()
        answer = self._return_object()
        end = datetime.now()
        runtime = (end-start).total_seconds()

        # Store both the object and runtime
        self.output().dump(answer, runtime=runtime)

class ListParameter(luigi.Parameter):

    _separator = ','

    def __init__(self, item_type=None, *args, **kwargs):
        self._item_type = item_type
        super(ListParameter, self).__init__(*args, **kwargs)

    def parse(self, x):
        items = x.split(',')
        if self._item_type is not None:
            items = map(self._item_type, items)
        return items

    def serialize(self, x):
        return ','.join(map(str, x))

class Model(MeansTask):
    """
    Creating this as a task, in case one needs to generate them
    """

    name = luigi.Parameter()
    _SUPPORTED_MODELS = {'p53': means.examples.MODEL_P53,
                         'hes1': means.examples.MODEL_HES1,
                         'dimerisation': means.examples.MODEL_DIMERISATION}

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

    def __init__(self, *args, **kwargs):
        super(MEAProblem, self).__init__(*args, **kwargs)

class TaskPreloadingHint(object):

    def preload(self):
        pass

class PreloadingWorker(luigi.worker.Worker):

    def _fork_task(self, children, task_id):
        task = self._Worker__scheduled_tasks[task_id]
        if isinstance(task, TaskPreloadingHint):
            task.preload()
        super(PreloadingWorker, self)._fork_task(children, task_id)


class Trajectory(MeansTask, TaskPreloadingHint):

    # All the parameters from MEAProblem, luigi does not support parametrised task hierarchies that well
    model_name = luigi.Parameter()
    max_order = luigi.IntParameter()
    closure = luigi.Parameter(default='scalar')
    multivariate = luigi.BooleanParameter(default=True)

    parameters = ListParameter(item_type=float)
    initial_conditions = ListParameter(item_type=float)
    timepoints_arange = ListParameter(item_type=float)

    # Solver kwargs, list the missing ones here with default=None
    solver = luigi.Parameter(default='ode15s')
    h = luigi.Parameter(default=None)

    def requires(self):
        return MEAProblem(model_name=self.model_name, max_order=self.max_order, closure=self.closure,
                          multivariate=self.multivariate)

    def _return_object(self):
        problem = self.input().load()



        logger.debug('Input: {0!r}'.format(self.input()))
        logger.debug(zip(self.input().get_cache().keys(), map(lambda x: hex(id(x)), self.input().get_cache().values())))

        try:
            logger.debug('!!!! {0} !!!!! Some memoised properties'.format(os.getpid(), problem._memoised_properties))
        except AttributeError:
            logger.debug('???? {0} !!!!! No memoised properties'.format(os.getpid()))

        timepoints = np.arange(*self.timepoints_arange)
        parameters = self.parameters
        initial_conditions = self.initial_conditions

        kwargs = {'solver': self.solver}
        if self.h is not None:
            kwargs['h'] = self.h
        simulation = means.Simulation(problem, **kwargs)
        return simulation.simulate_system(parameters, initial_conditions, timepoints)

    def preload(self):
        if self.input().exists():
            logger.warn('Preloading {0} {1}'.format(self.__class__.__name__, hex(id(self))))
            start = datetime.now()
            problem = self.input().load()
            problem.right_hand_side_as_function
            end = datetime.now()
            logger.warn('Took: {0}'.format((end-start).total_seconds()))

class LotsOfTrajectoriesTask(MeansTask):

    def requires(self):
        model_name = 'p53'
        max_order = 3

        parameters = []
        for c_2 in np.arange(1.5, 2, 0.1):
            parameters.append([90, 0.002, c_2, 1.1, 0.93, 0.96, 0.01])

        initial_conditions = [70, 30, 60]
        timepoints = [0, 40, 0.1]

        return [Trajectory(model_name=model_name, max_order=max_order, parameters=x,
                           initial_conditions=initial_conditions, timepoints_arange=timepoints) for x in parameters]

    def _return_object(self):

        inputs = self.input()

        return [i.load() for i in inputs]

class PreloadingWorkerSchedulerFactory(luigi.interface.WorkerSchedulerFactory):
    def create_worker(self, scheduler, worker_processes):
        return PreloadingWorker(scheduler=scheduler, worker_processes=worker_processes)

if __name__ == '__main__':
    luigi.run(worker_scheduler_factory=PreloadingWorkerSchedulerFactory())
    #luigi.run()