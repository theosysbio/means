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
        param_values = [getattr(self, x[0]) for x in params if x[1].significant]

        to_filesafe_string = lambda x: str(x).replace(',', '_').replace(' ', '_')
        params_str = '-'.join(map(to_filesafe_string, param_values))
        if params_str:
            params_str = ''.join(['-', params_str])
        return '{0}{1}{2}'.format(class_, params_str, self._file_extension)

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

    figure_format = luigi.Parameter(default='pdf')

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
        answer.savefig(self.output().path, bbox_inches='tight')

        # Close the figure
        plt.close(answer)

class TexFigureTask(Task):
    """
    Creates a latex figure that joins one or more figures using `LaTeX subfloats`_.

    Returns a .tex file with the figure in LaTeX notation that can then be included to other files

    .. `LaTeX subfloats`: https://en.wikibooks.org/wiki/LaTeX/Floats,_Figures_and_Captions#Subfloats
    """

    label = luigi.Parameter()
    caption = luigi.Parameter(significant=False)
    placement = luigi.Parameter(default='tb', significant=False)
    number_of_columns = luigi.Parameter(default=0)
    standalone = luigi.BooleanParameter(default=False)

    @property
    def _file_extension(self):
        return '.tex'

    def _output(self):
        output = luigi.File(self.filepath)
        return output

    def _return_object(self):
        files = []
        for figure in self.input():
            files.append(figure.path)
        return files

    def _store_output_and_runtime(self, answer, runtime):

        template = r'''
        {standalone}\documentclass{{article}}
        {standalone}\usepackage{{graphicx}}
        {standalone}\usepackage{{caption}}
        {standalone}\usepackage{{subcaption}}
        {standalone}\begin{{document}}
        \begin{{figure}}
            \centering
            {subfigures}
            \caption{{{caption}}}
            \label{{fig:{label}}}
        \end{{figure}}
        {standalone}\end{{document}}
        '''

        subfigures_template = r'''
        \begin{{subfigure}}[b]{{{width}\textwidth}}
            \includegraphics[width=\textwidth]{{{figure_file}}}
            \caption{{}} % TODO: caption
        \end{{subfigure}}
        '''

        number_of_subfigures = len(answer)
        maxwidth = 0.9  # Proportion of the total \textwidth
        number_of_columns = self.number_of_columns

        if number_of_columns <= 0:
            number_of_columns = number_of_subfigures

        width_per_column = round(maxwidth / number_of_columns, 2)
        subfigure_strs = []
        for subfigure in answer:
            filename, extension = os.path.splitext(subfigure)
            # Change filename from smth.pdf to {smth}.pdf -- otherwise LaTeX is not happy
            masked_filename = '{{{0}}}{1}'.format(filename, extension)
            subfigure_strs.append(subfigures_template.format(width=width_per_column,
                                                             figure_file=masked_filename))


        with self.output().open('w') as f:
            f.write(template.format(caption=self.caption,
                                    label=self.label,
                                    subfigures='~'.join(subfigure_strs),
                                    standalone='' if self.standalone else '%'))







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

class TrajectoryTaskBase(Task):

    model_name = luigi.Parameter()

    # General parameters for trajectory
    parameters = ListParameter(item_type=float)
    initial_conditions = ListParameter(item_type=float)
    timepoints_arange = ListParameter(item_type=float)


    def _simulation_object(self):
        raise NotImplementedError

    def _return_object(self):

        timepoints = np.arange(*self.timepoints_arange)
        parameters = self.parameters
        initial_conditions = self.initial_conditions

        simulation = self._simulation_object()

        try:
            return simulation.simulate_system(parameters, initial_conditions, timepoints)
        except means.SolverException as e:
            # We want to return the exception if it is legitimate failure of solver - i.e. the task has succeeded
            # but the solver did not
            return e
        except Exception:
            # Any other exception is still raised as that means the task failed
            raise

class TrajectoryTask(TrajectoryTaskBase, TaskPreloadingHint):

    # All the parameters from MEAProblem, luigi does not support parametrised task hierarchies that well
    max_order = luigi.IntParameter()
    closure = luigi.Parameter(default='scalar')
    multivariate = luigi.BooleanParameter(default=True)

    # Solver kwargs, list the missing ones here with default=None
    solver = luigi.Parameter(default='ode15s')
    h = luigi.Parameter(default=None)

    def requires(self):
        return MEATask(model_name=self.model_name, max_order=self.max_order, closure=self.closure,
                       multivariate=self.multivariate)

    def _simulation_object(self):

        problem = self.input().load()

        kwargs = {'solver': self.solver}
        if self.h is not None:
            kwargs['h'] = self.h
        simulation = means.Simulation(problem, **kwargs)

        return simulation


    def preload(self):
        if self.input().exists():
            logger.debug('Preloading {0} {1}'.format(self.__class__.__name__, hex(id(self))))
            # Cache the load from file
            problem = self.input().load()
            # Cache the right_hand_side_as_function
            __ = problem.right_hand_side_as_function


class SSATrajectoryTask(TrajectoryTaskBase):

    n_simulations = luigi.IntParameter()

    def requires(self):
        return ModelTask(name=self.model_name)

    def _simulation_object(self):

        problem = means.StochasticProblem(self.input().load())
        simulation = means.SSASimulation(problem, n_simulations=self.n_simulations)

        return simulation