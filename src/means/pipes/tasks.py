import re
import luigi
import os
from means.pipes.interface import TaskPreloadingHint
from means.pipes.parameters import ListParameter, ModelParameter, ListOfKeyValuePairsParameter
from means.pipes.targets import PickleSerialiserWithAdditionalParameters
from datetime import datetime
import numpy as np
from means.util.logs import get_logger
import means

# Allow getting the output directory from [output] > directory in config file
OUTPUT_DIR = luigi.configuration.get_config().get('output', 'directory', 'task-output')

logger = get_logger(__name__)


def filesafe_string(original_filename):
    """
    Replaces all non ascii or non-digit characters
    :param original_filename: Original string replace
    :return: version of the original_filename that is safe as a filename
    """
    original_filename = str(original_filename)
    # Based on http://stackoverflow.com/a/295466/171400
    # change all non-safe symbols to _
    original_filename = re.sub(r'[^\d\w.-]', '_', original_filename)
    # Change multiple spaces to single space
    original_filename = re.sub(r'_+', '_', original_filename)
    # Remove spaces at the start and end of the string
    original_filename = original_filename.strip('_')
    return original_filename

class TaskBase(luigi.Task):
    """
    Base class for all tasks in :mod:`means.pipes`

    """

    _FILENAME_LENGTH_LIMIT = 200
    """Maximum supported length for filenames. Hashed filenames will be used instead of human-readable ones if
    the length goes above that. This needs to be a bit lower, as luigi appends some temp extension while saving files"""

    use_human_readable_filenames = True
    """
    If set to true, human-readable filenames will try to be generated,
    otherwise, the parameter string will be shortened using md5. A clash is less likely, if non-human readable filenames
    used
    """

    def __init__(self, *args, **kwargs):
        super(TaskBase, self).__init__(*args, **kwargs)

    @property
    def _file_extension(self):
        """
        The file extension for the serialised objects to use.

        :return: the file extension, including the dot in front of it. i.e. ``.pickle``
        """
        return ''

    @property
    def _filename(self):
        """
        Method that generates the filename of the class.
        This defaults to the class name, followed by dash-separated parameters of the class followed by
        :attr:`TaskBase._file_extension`

        :return: The generated filename
        :rtype: str
        """
        try:
            return self.__filename
        except AttributeError:
            pass

        # Default filename that just lists class name and parameters in dashes
        class_ = self.__class__.__name__
        filename = ''

        params = self.get_params()
        param_values = [getattr(self, x[0]) for x in params if x[1].significant]

        if self.use_human_readable_filenames:
            params_str = '-'.join(map(filesafe_string, param_values))
            if params_str:
                params_str = ''.join(['-', params_str])

            params_str = params_str.strip('-')

            filename = '{0}-{1}{2}'.format(class_, params_str, self._file_extension)

        if not self.use_human_readable_filenames or len(filename) > self._FILENAME_LENGTH_LIMIT:
            import hashlib
            params_str = hashlib.md5(';'.join(map(str, param_values))).hexdigest()
            filename = '{0}-{1}{2}'.format(class_, params_str, self._file_extension)

        assert(filename != '')
        # Cache the filename before returning, especially important for the hashlib generated ones
        self.__filename = filename

        return filename




    def _output(self):
        """
        A function that returns the class output by default.

        If you want to change the behaviour of this function, please override it, rather than the actual
        :meth:`TaskBase.output()` method
        """
        raise NotImplementedError

    def output(self):
        """
        Returns the output of the class.
        Ensures we only have one instance of the output object, meaning we have only one cache per output object.

        Please do not override this in your class, and override :meth:`TaskBase._output()` instead.

        :return: Cached result of :meth:`TaskBase._output()`
        """
        try:
            # Make sure we have only one instance of output object
            return self.__output
        except AttributeError:
            output = self._output()
            self.__output = output
            return output

    @property
    def filepath(self):
        """
        Generates the filepath of the task, default is of format ``<OUTPUT_DIR>/<CLASS>/<FILENAME>``
        where ``<FILENAME>`` is defined in :attr:`TaskBase._filename`.

        :return:
        """
        # Force the directory structure <OUTPUT_DIR>/<CLASS>/<FILENAME>
        return os.path.join(OUTPUT_DIR,
                            self.__class__.__name__,
                            self._filename)


    def _return_object(self):
        """
        Implementation of the main logic in the task.
        This function should perform the task, generate the output and return it.

        The rest of the class will handle storing that output.

        e.g. ::

            >>> from means.pipes import Task
            >>> class MyTask(Task):
            ...     def _return_object(self):
            ...         a = 'foo'
            ...         b = 'bar'
            ...
            ...         # This would store `a` and `b` as a tuple in the output file `<output_dir>/MyTask/MyTask.pickle'
            ...         return a, b

        """
        raise NotImplementedError

    def _store_output_and_runtime(self, answer, runtime):
        """
        Implements storage of the output from the task and runtime of the task into some file.

        :param answer: The answer returned by :meth:`TaskBase._return_object()`
        :param runtime: Runtime of the task in seconds
        :type runtime: float
        """
        raise NotImplementedError

    def run(self):
        """
        Runs the specified task and keep track of the runtime

        Subclasses should not override this method directly, and should override the
        :meth:`TaskBase._return_object()` instead.
        """
        # Poor man's timing
        start = datetime.now()
        answer = self._return_object()
        end = datetime.now()
        runtime = (end-start).total_seconds()
        self._store_output_and_runtime(answer, runtime)



class Task(TaskBase):
    """
    A wrapper around :class:`luigi.Task` that provides some basic functionality used in means.

    Namely, it defines :class:`~means.interface.PickleSerialiserWithAdditionalParameters` as an output source
    and automatically sets file extension to ``.pickle``.

    The classes inheriting from this class should define their own :meth:`~Task._return_object()` implementation that
    would perform the required task and return the object that is serialised.

    For example::
        >>> from means.pipes import Task
        >>> class MyTask(Task):
        ...     def _return_object(self):
        ...         a = 'foo'
        ...         b = 'bar'
        ...
        ...         # This would store `a` and `b` as a tuple in the output file `<output_dir>/MyTask/MyTask.pickle'
        ...         return a, b

    """

    @property
    def _file_extension(self):
        return '.pickle'

    def _return_object(self):
        """
        Implementation of the main logic in the task.
        This function should perform the task, generate the output and return it.

        The rest of the class will handle storing that output.

        """
        raise NotImplementedError

    def _store_output_and_runtime(self, answer, runtime):
        # Store both the object and runtime
        self.output().dump(answer, runtime=runtime)

    def _output(self):
        output = PickleSerialiserWithAdditionalParameters(self.filepath)
        return output

class FigureTask(TaskBase):
    """
    Class that can store :mod:`matplotlib` figures in pdf/svg format.

    To use it define a class that would override the :meth:`FigureTask._return_object` so it returns a
    :class:`matplotlib.Figure`, which then would be saved in the format specified by :attr:`FigureTask.figure_format`.

    For example ::

         >>> from means.pipes import FigureTask
         >>> class MyFigure(FigureTask):
         ...
         ...     def _return_object(self):
         ...         from matplotlib import pyplot as plt
         ...         # Create a figure
         ...         fig = plt.figure()
         ...
         ...         # Plot something
         ...         ax = plt.subplot(1,1,1)
         ...         ax.plot([1,2], [3,4], label='foobar')
         ...         fig.legend()
         ...
         ...         # just return the figure, do not issue ``plt.show()`` or anything of that sort
         ...         return fig
    """

    figure_format = luigi.Parameter(default='pdf')
    """Figure format to render the figure in. PDF by default. Can also support SVG"""

    def _return_object(self):
        """
        Implementation of the main logic in the task.
        This function should perform the task, generate the output and return it.

        The rest of the class will handle storing that output.
        """
        raise NotImplementedError

    @property
    def _file_extension(self):
        """
        The figure extension

        :return: :attr:`FigureTask.figure_format` preceded by a dot, i.e. ``".pdf"``
        """
        return '.{0}'.format(self.figure_format)

    def _output(self):
        """

        :return: Output object pointing to the file where the figure is saved
        """
        output = luigi.File(self.filepath)
        return output

    def _store_output_and_runtime(self, answer, runtime):
        """
        Renders the figure into the file specified by :attr:`FigureTask._output`
        """
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
    Creates a LaTeX figure that joins one or more figures using `LaTeX subfloats`_.
    These figures need to be specified in the :meth:`TexFigureTask.requires()` object.

    Saves the resulting figure into a .tex file that can be included in other files or compiled to pdf (if standalone
    is set)

    Example usage::

        >>> class MyFigure(Figure):
        ...    name = Parameter()
        ...
        ... class MyTexFigure(TexFigureTask):
        ...     label = 'my-tex-figure'  # Set the label of figure
        ...     caption = 'Some Caption' # Set it's caption
        ...     standalone = True        # Make it standalone
        ...     number_of_columns = 1    # Allow only one column (meaning the two figures will be in two rows)
        ...
        ...     def requires(self):
        ...         # Specify all the figures as dependancies only, the package will do the rest
        ...         return [MyFigure(name='foo'), MyFigure(name='bar')]


    .. `LaTeX subfloats`: https://en.wikibooks.org/wiki/LaTeX/Floats,_Figures_and_Captions#Subfloats
    """

    label = luigi.Parameter()
    """Label of the LaTeX figure"""

    caption = luigi.Parameter(significant=False)
    """Caption of the figure"""

    placement = luigi.Parameter(default='tb', significant=False)
    """Placement of the figure, defaults to ``tb``"""

    number_of_columns = luigi.Parameter(default=0)
    """Number of columns to structure the subfloats to. If set to zero, will put the figures into one row"""

    standalone = luigi.BooleanParameter(default=False)
    """If set to true, generates a standalone tex file, that can be compiled immediately, otherwise
       generates a file that needs to be included in some document"""

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

class MEATask(Task):
    """
    Task to perform MEA Approximation and return it's result.
    """

    model = ModelParameter()
    """Model to use"""

    max_order = luigi.IntParameter()
    """MAX order to perform the approximation"""

    closure = luigi.Parameter(default='scalar')
    """Closure method to use"""

    multivariate = luigi.BooleanParameter(default=True)
    """Whether to use multivariate or univariate closure (where available)"""


    def _return_object(self):
        model = self.model

        # Scalar closure currently does not support univariate/multivariate
        if self.closure != 'scalar':
            kwargs = {'multivariate': self.multivariate}
        else:
            kwargs = {}
        problem = means.mea_approximation(model, self.max_order, closure=self.closure,
                                          **kwargs)

        return problem

class TrajectoryTaskBase(Task):
    """
    Base-class for trajectories.
    """

    model = ModelParameter()
    """Model name to use"""

    # General parameters for trajectory
    parameters = ListParameter(item_type=float)
    """Parameters to simulate trajectories for"""

    initial_conditions = ListParameter(item_type=float)
    """Initial conditions to use"""

    timepoints_arange = ListParameter(item_type=float)
    """An arangement of the timepoints to simulate,
       e.g. ``(0, 40, 0.1)`` would simulate from 0 to 40 in 0.1 increments"""


    def _simulation_object(self):
        """
        A method that sub-classes should override, this method should return a simulation object
        that would have ``.simulate_system()`` method, e.g. :class:`~means.simulation.simulate.Simulation`
        """
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
    """
    Task to simulate the system, and return the resulting trajectories.
    Uses the standard Simulation object, :class:`means.simulation.simulate.Simulation`.

    See also :class:`~means.pipes.tasks.SSATrajectoryTask` for SSA simulation support.
    """

    # All the parameters from MEAProblem, luigi does not support parametrised task hierarchies that well
    max_order = MEATask.max_order
    """Maximum order of MEA approximation to use"""
    closure = MEATask.closure
    """Closure method to use"""
    multivariate = MEATask.multivariate
    """Use multivariate closure (where available)"""

    # Solver kwargs, list the missing ones here with default=None
    solver = luigi.Parameter(default='ode15s')
    """ODE solver to use, defaults to ode15s"""

    solver_kwargs = ListOfKeyValuePairsParameter(default=[])
    """Keyword arguments to pass to solver"""

    def requires(self):
        return MEATask(model=self.model, max_order=self.max_order, closure=self.closure,
                       multivariate=self.multivariate)

    def _simulation_object(self):

        problem = self.input().load()
        simulation = means.Simulation(problem, solver=self.solver, **dict(self.solver_kwargs))

        return simulation


    def preload(self):
        if self.input().exists():
            logger.debug('Preloading {0} {1}'.format(self.__class__.__name__, hex(id(self))))
            # Cache the load from file
            problem = self.input().load()
            # Cache the right_hand_side_as_function
            __ = problem.right_hand_side_as_function


class SSATrajectoryTask(TrajectoryTaskBase):
    """
    Generates a SSA trajectory for the particular set of parameters.
    See :class:`~means.simulation.ssa.SSASimulation` for more details.

    See also :class:`~means.pipes.tasks.TrajectoryTask` for ODE simulation support.

    """

    n_simulations = luigi.IntParameter()
    """Number of simulations to use in SSA"""

    def _simulation_object(self):

        problem = means.StochasticProblem(self.model)
        simulation = means.SSASimulation(problem, n_simulations=self.n_simulations)

        return simulation