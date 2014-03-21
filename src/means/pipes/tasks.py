import luigi
import os
from means.pipes.interface import TaskPreloadingHint
from means.pipes.parameters import ListParameter
from means.pipes.targets import PickleSerialiserWithAdditionalParameters
from datetime import datetime
import numpy as np
from means.util.logs import get_logger
import means.examples
import means

# Allow getting the output directory from [output] > directory in config file
OUTPUT_DIR = luigi.configuration.get_config().get('output', 'directory', 'task-output')

logger = get_logger(__name__)

class TaskBase(luigi.Task):

    @property
    def _file_extension(self):
        return ''

    @property
    def _filename(self):
        # Default filename that just lists class name and parameters in dashes
        class_ = self.__class__.__name__
        params = self.get_params()
        param_values = [getattr(self, x[0]) for x in params]
        params_str = '-'.join(map(str, param_values))
        return '{0}-{1}{2}'.format(class_, params_str, self._file_extension)

    def _output(self):
        raise NotImplementedError

    def output(self):
        try:
            # Make sure we have only one instance of output object
            return self.__output
        except AttributeError:
            output = self._output()
            self.__output = output
            return output

    @property
    def filepath(self):
        # Force the directory structure <OUTPUT_DIR>/<CLASS>/<FILENAME>
        return os.path.join(OUTPUT_DIR,
                            self.__class__.__name__,
                            self._filename)


    def _return_object(self):
        raise NotImplementedError

    def _store_output_and_runtime(self, answer, runtime):
        raise NotImplementedError

    def run(self):
        # Poor man's timing
        start = datetime.now()
        answer = self._return_object()
        end = datetime.now()
        runtime = (end-start).total_seconds()
        self._store_output_and_runtime(answer, runtime)



class Task(TaskBase):
    """
    A wrapper around luigi task that would automatically set the output variable to a standard used in MEANS pipelines
    """

    @property
    def _file_extension(self):
        return '.pickle'


    def _store_output_and_runtime(self, answer, runtime):
        # Store both the object and runtime
        self.output().dump(answer, runtime=runtime)

    def _output(self):
        output = PickleSerialiserWithAdditionalParameters(self.filepath)
        return output

class FigureTask(TaskBase):

    figure_format = luigi.Parameter(default='svg')

    @property
    def _file_extension(self):
        return '.{0}'.format(self.figure_format)

    def _output(self):
        output = luigi.File(self.filepath)
        return output

    def _store_output_and_runtime(self, answer, runtime):
        from matplotlib import pyplot as plt
        assert (isinstance(answer, plt.Figure))

        # Equivalent to unix touch command - ensure file is writable
        f = self.output().open('w')
        f.close()
        # Write to filename, not file object, as matplotlib doesn't like the latter
        answer.savefig(self.output().path)
        # Close the figure
        plt.close(answer)


class ModelTask(Task):
    """
    Return a model from one of the predefined models
    """
    name = luigi.Parameter()
    _SUPPORTED_MODELS = {'p53': means.examples.MODEL_P53,
                         'hes1': means.examples.MODEL_HES1,
                         'dimerisation': means.examples.MODEL_DIMERISATION,
                         'michaelis-menten': means.examples.MODEL_MICHAELIS_MENTEN,
                         'lotka-volterra': means.examples.MODEL_LOTKA_VOLTERRA}

    def _return_object(self):
        return self._SUPPORTED_MODELS[self.name]

class MEATask(Task):
    """
    Task to perform MEA Approximation and return result
    """

    model_name = luigi.Parameter()
    max_order = luigi.IntParameter()
    closure = luigi.Parameter()
    multivariate = luigi.BooleanParameter(default=True)

    def requires(self):
        return ModelTask(self.model_name)

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

class TrajectoryTask(Task, TaskPreloadingHint):

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
        return MEATask(model_name=self.model_name, max_order=self.max_order, closure=self.closure,
                       multivariate=self.multivariate)

    def _return_object(self):
        problem = self.input().load()

        timepoints = np.arange(*self.timepoints_arange)
        parameters = self.parameters
        initial_conditions = self.initial_conditions

        kwargs = {'solver': self.solver}
        if self.h is not None:
            kwargs['h'] = self.h
        simulation = means.Simulation(problem, **kwargs)
        try:
            return simulation.simulate_system(parameters, initial_conditions, timepoints)
        except means.SolverException as e:
            # We want to return the exception if it is legitimate failure of solver - i.e. the task has succeeded
            # but the solver did not
            return e
        except Exception:
            # Any other exception is still raised as that means the task failed
            raise


    def preload(self):
        if self.input().exists():
            logger.debug('Preloading {0} {1}'.format(self.__class__.__name__, hex(id(self))))
            # Cache the load from file
            problem = self.input().load()
            # Cache the right_hand_side_as_function
            __ = problem.right_hand_side_as_function