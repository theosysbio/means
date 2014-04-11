"""
Generates the hit-and-miss figures for the report from scratch.

To run it, first run a luigi daemon on one terminal window:

    bash$ luigid

Then on, another window, start this script

    bash$ python figure_hit_and_miss.py --workers 8

Change the number of workers to the number of CPUs on your computer.
Open your browser to http://127.0.0.1:8042 to see the luigi task visualiser an follow the progress there.

Known issues:

    It is likely that matplotlib will fail when multiple workers are used.
    If that happens, the main worker will die and all `FigureTask`s will be marked as red in the luigi visualiser.
    This can be mitigated by re-running this script with only one worker (so no threading is used)
    What should run only the failed tasks and regenerate all the figures.


"""
import scipy.interpolate
from means import TrajectoryCollection, SolverException
from means.pipes import *
import numpy as np
import itertools
from collections import namedtuple

from means.util.logs import get_logger
logger = get_logger(__name__)

_ParameterSet = namedtuple('ParameterSet', ['parameters', 'initial_conditions', 'timepoints_arange',
                                           'marker', 'label'])

class ParameterSet(_ParameterSet):

    # Make the printing clean
    def __str__(self):
        return self.marker.replace('$', '').replace('\\', '')

INTERESTING_PARAMETER_SETS = [ParameterSet(parameters=[90.0, 0.002, 2.5, 1.1, 1.8, 0.96, 0.01],
                                           initial_conditions=[70.0, 30.0, 60.0],
                                           timepoints_arange=[0.0, 40.0, 0.1],
                                           marker='$\\times$', label='$c_2=2.5$ and $c_4=1.8$'),
                              ParameterSet(parameters=[90.0, 0.002, 2.0, 1.1, 1.8, 0.96, 0.01],
                                           initial_conditions=[70.0, 30.0, 60.0],
                                           timepoints_arange=[0.0, 40.0, 0.1],
                                           marker='$\\ast$', label='$c_2=2.0$ and $c_4=1.8$'),
                              ParameterSet(parameters=[90.0, 0.002, 1.6, 1.1, 2.1, 0.96, 0.01],
                                           initial_conditions=[70.0, 30.0, 60.0],
                                           timepoints_arange=[0.0, 40.0, 0.1],
                                           marker='$\\bullet$', label='$c_2=1.6$ and $c_4=2.1$'),
                              ParameterSet(parameters=[90.0, 0.002, 2.4, 1.1, 0.7, 0.96, 0.01],
                                           initial_conditions=[70.0, 30.0, 60.0],
                                           timepoints_arange=[0.0, 40.0, 0.1],
                                           marker='$\\blacksquare$', label='$c_2=2.3$ and $c_4=0.7$'),
                              ParameterSet(parameters=[90.0, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01],
                                           initial_conditions=[70.0, 30.0, 60.0],
                                           timepoints_arange=[0.0, 40.0, 0.1],
                                           marker='$\\star$', label='$c_2=1.7$ and $c_4=0.93$, as in original paper'),
                              ]

class P53Model(means.Model):
    """
    A wrapper around means. Model that initialises it with P53 parameters and changes the __str__ function to
    print a shorter string
    """
    def __init__(self):
        from means.examples import MODEL_P53
        super(P53Model, self).__init__(MODEL_P53.species, MODEL_P53.parameters, MODEL_P53.propensities,
                                       MODEL_P53.stoichiometry_matrix)

    def __str__(self):
        # Override the str() methods so they do not print the whole blerch of things, but
        # only a nice and easily readable "p53"
        return 'p53'

class HitAndMissDataParametersMixin(object):

    max_order = TrajectoryTask.max_order
    timepoints_arange = TrajectoryTask.timepoints_arange
    number_of_ssa_simulations = SSATrajectoryTask.n_simulations
    point_sparsity = FloatParameter(default=0.1)

    solver = TrajectoryTask.solver
    solver_kwargs = TrajectoryTask.solver_kwargs

    closure = TrajectoryTask.closure
    multivariate = TrajectoryTask.multivariate


class FigureHitAndMissData(HitAndMissDataParametersMixin, Task):

    def x_values(self):
        return np.arange(1.5, 2.7, self.point_sparsity)

    def y_values(self):
        return np.arange(0.6, 2.5, self.point_sparsity)

    def x_param(self):
        return 'c_2'

    def y_param(self):
        return 'c_4'

    def x_param_index(self):
        return 2

    def y_param_index(self):
        return 4

    def requires(self):
        model = P53Model()
        max_order = self.max_order

        initial_conditions = [70.0, 30.0, 60.0]

        parameters = []
        for x in self.x_values():
            for y in self.y_values():
                full_params = [90.0, 0.002, 1, 1.1, 1, 0.96, 0.01]
                full_params[self.x_param_index()] = round(x, 6)
                full_params[self.y_param_index()] = round(y, 6)

                parameters.append(full_params)

        # We want to specify all the trajectories we need to compute as requirements of this task,
        # so luigi handles their execution and scheduling, not us.
        regular_trajectories = [TrajectoryTask(model=model, max_order=max_order,
                                parameters=x,
                                initial_conditions=initial_conditions, timepoints_arange=self.timepoints_arange,
                                solver=self.solver,
                                solver_kwargs=self.solver_kwargs,
                                closure=self.closure,
                                multivariate=self.multivariate)
                                for x in parameters]

        ssa_trajectories = [SSATrajectoryTask(model=model,
                                              parameters=x,
                                              initial_conditions=initial_conditions,
                                              timepoints_arange=self.timepoints_arange,
                                              n_simulations=self.number_of_ssa_simulations)
                                              for x in parameters]

        return regular_trajectories + ssa_trajectories

    @classmethod
    def distance_between_trajectories(self, lookup, trajectory_collection):
        distance = 0.0
        for trajectory_a in trajectory_collection:
            try:
                trajectory_b = lookup[trajectory_a.description]
            except KeyError:
                continue

            distance += np.sum(np.square(trajectory_a.values - trajectory_b.values))

        return distance

    def _return_object(self):

        ssa_trajectory_lookup = {}

        for task, trajectory_buffer in itertools.izip(self.requires(), self.input()):
            if isinstance(task, SSATrajectoryTask):
                lookup = {}
                for trajectory in trajectory_buffer.load():
                    lookup[trajectory.description] = trajectory

                ssa_trajectory_lookup[tuple(task.parameters)] = lookup

        x_values, y_values = self.x_values(), self.y_values()
        x_value_lookup = {round(value, 6): i for i, value in enumerate(x_values)}
        y_value_lookup = {round(value, 6): i for i, value in enumerate(y_values)}

        # Remember that matrices are indices rows first, then columns
        # meaning index_y first, index_x afterwards
        distance_grid = np.empty((len(y_values), len(x_values)), dtype=np.float)
        distance_grid.fill(np.nan)

        failure_times_grid = np.empty((len(y_values), len(x_values)), dtype=np.float)
        failure_times_grid.fill(np.nan)

        failure_mask = np.empty((len(y_values), len(x_values)), dtype=bool)
        failure_mask.fill(False)

        for task, trajectory_buffer in itertools.izip(self.requires(), self.input()):
            if isinstance(task, SSATrajectoryTask):
                continue

            x_val, y_val = round(task.parameters[2], 6), round(task.parameters[4], 6)
            try:
                index_x, index_y = x_value_lookup[x_val], y_value_lookup[y_val]
            except KeyError:
                raise

            trajectory = trajectory_buffer.load()
            if isinstance(trajectory, SolverException):
                distance_grid[index_y, index_x] = np.nan
                base_exception = trajectory.base_exception
                try:
                    failure_time = base_exception.t
                except AttributeError:
                    # Some solvers, i.e. RODAS do not give the time at failure.
                    failure_time = np.nan
                failure_times_grid[index_y, index_x] = np.nan
                failure_mask[index_y, index_x] = True
            elif isinstance(trajectory, TrajectoryCollection):
                ssa_equivalent = ssa_trajectory_lookup[tuple(task.parameters)]
                distance = FigureHitAndMissData.distance_between_trajectories(ssa_equivalent, trajectory)
                # We sometimes go off the scale
                if np.isnan(distance):
                    distance = float('inf')
                distance_grid[index_y, index_x] = distance
            else:
                raise Exception('Got {0!r} as trajectory, expected either SolverException'
                                ' or TrajectoryCollection'.format(trajectory))

        return x_values, y_values, distance_grid, failure_mask, failure_times_grid, self.x_param(), self.y_param()


# noinspection PyArgumentList
class FigureHitAndMiss(HitAndMissDataParametersMixin, FigureTask):

    draw_interesting_parameters = Parameter(significant=False, default=True)

    def requires(self):

        # Let's make sure self.max_order is always first
        data = FigureHitAndMissData(max_order=self.max_order,
                                    timepoints_arange=self.timepoints_arange,
                                    number_of_ssa_simulations=self.number_of_ssa_simulations,
                                    point_sparsity=self.point_sparsity,
                                    solver=self.solver,
                                    solver_kwargs=self.solver_kwargs,
                                    closure=self.closure,
                                    multivariate=self.multivariate)

        return data

    def _return_object(self):
        from matplotlib import pyplot as plt
        from matplotlib import colors
        from matplotlib.cm import jet
        fig = plt.figure()
        ax = fig.gca()

        data = self.input().load()
        levels = [10, 100, 500, 1000, 5000, 1e4, 1e5, 1e10, 1e20, 1e30, 1e40, 1e50, 1e60, 1e70, 1e80, 1e90, 1e100]

        cmap = jet
        cmap.set_bad('k', 1)
        ax.patch.set_hatch('//')
        x_values, y_values, distance_grid, failure_mask, failure_times_grid, x_param, y_param = data
        distance_grid[np.isinf(distance_grid)] = 1e100  # Contour plots cannot handle infinities
        distance_grid = np.ma.array(distance_grid, mask=failure_mask)



        ax.contourf(x_values, y_values, distance_grid,
                    levels, norm=colors.LogNorm(),
                    vmax=1e10, vmin=1, cmap=cmap)
        contours = ax.contour(x_values, y_values, distance_grid,
                              [10, 100, 500, 1000, 5000, 1e4, 1e5, 1e10, 1e20, 1e40], colors='k')
        ax.clabel(contours, fmt="%.5g")
        ax.set_xlabel('${0}$'.format(x_param))
        ax.set_ylabel('${0}$'.format(y_param))

        if not self.solver_kwargs:
            solver_kwargs_string = ''
        else:
            solver_kwargs_string = ' ({0})'.format(';'.join(['{0}: {1}'.format(key, value)
                                                             for key, value in self.solver_kwargs]))
        ax.set_title('Solver: {0}{1}, max_order: {2}'.format(self.solver,
                                                      solver_kwargs_string,
                                                      self.max_order))

        if self.draw_interesting_parameters:
            x_param_index = self.requires().x_param_index()
            y_param_index = self.requires().y_param_index()

            for param_set in INTERESTING_PARAMETER_SETS:
                params = param_set.parameters
                x = params[x_param_index]
                y = params[y_param_index]
                ax.plot([x], [y], color='k', ms=10, marker=param_set.marker, label=param_set.label)

        return fig

class FigureSSAvMEATrajectory(FigureTask):

    model = TrajectoryTask.model

    timepoints_arange = TrajectoryTask.timepoints_arange
    parameters = TrajectoryTask.parameters
    initial_conditions = TrajectoryTask.initial_conditions

    max_order = TrajectoryTask.max_order
    closure = TrajectoryTask.closure
    multivariate = TrajectoryTask.multivariate

    solver = TrajectoryTask.solver
    solver_kwargs = TrajectoryTask.solver_kwargs

    number_of_ssa_simulations = SSATrajectoryTask.n_simulations
    title = Parameter(default=None, significant=False)

    def requires(self):

        ssa_trajectory = SSATrajectoryTask(model=self.model,
                                           parameters=self.parameters,
                                           initial_conditions=self.initial_conditions,
                                           timepoints_arange=self.timepoints_arange,
                                           n_simulations=self.number_of_ssa_simulations,)

        trajectory = TrajectoryTask(model=self.model, max_order=self.max_order,
                                    parameters=self.parameters,
                                    initial_conditions=self.initial_conditions,
                                    timepoints_arange=self.timepoints_arange,
                                    solver=self.solver,
                                    solver_kwargs=self.solver_kwargs,
                                    closure=self.closure,
                                    multivariate=self.multivariate)

        return [ssa_trajectory, trajectory]

    def _return_object(self):
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import FormatStrFormatter

        figure = plt.figure(figsize=(5,5))

        ssa_trajectory_buffer, trajectory_buffer = self.input()
        ssa_trajectories = ssa_trajectory_buffer.load()
        ssa_trajectories = [ssa_trajectories[0]] # Let's take only the first trajectory
        trajectories = trajectory_buffer.load()
        number_of_ssa_trajectories = len(ssa_trajectories)

        fmt = FormatStrFormatter('%.3g')
        if isinstance(trajectories, TrajectoryCollection):
            for i, (ssa_trajectory, trajectory) in enumerate(zip(ssa_trajectories, trajectories)):
                assert(ssa_trajectory.description == trajectory.description)

                ax = plt.subplot(1, number_of_ssa_trajectories, i+1)
                ax.yaxis.set_major_formatter(fmt)
                if self.title is not None:
                    plt.title(self.title)
                else:
                    plt.title(ssa_trajectory.description.mathtext())

                ssa_trajectory.plot(label='SSA', color='b')
                trajectory.plot(label='Solver', color='r')
                plt.legend()

                sum_of_sqr_distance = np.sum(np.square(ssa_trajectory.values - trajectory.values))
                plt.annotate('Distance={0:.2f}'.format(sum_of_sqr_distance), xy=(1, 0),
                             xycoords='axes fraction', fontsize=16, xytext=(-5, 5),
                             textcoords='offset points', ha='right', va='bottom')

                # plt.subplot(2, number_of_ssa_trajectories, i+1+number_of_ssa_trajectories)
                # plt.title(ssa_trajectory.description.mathtext() + ' - difference')
                # (ssa_trajectory - trajectory).plot(label='difference')
        elif isinstance(trajectories, SolverException):
            for i, ssa_trajectory in enumerate(ssa_trajectories):

                plt.subplot(1, number_of_ssa_trajectories, i+1)

                if self.title is not None:
                    plt.title(self.title)
                else:
                    plt.title(ssa_trajectory.description.mathtext())

                ssa_trajectory.plot(label='SSA')
                plt.plot([], [], label='Solver (failed)')

                plt.legend()
        else:
            raise Exception("Got unexpected kind of trajectories output: {0!r}".format(trajectories))

        plt.xlabel('Time')
        plt.ylabel('Concentration')

        return figure

class FigureHitAndMissTex(TexFigureTask):
    """
    Join all hit and miss figures into one tex file
    """

    # Note that this is not a parameter, it is a constant
    timepoints_arange = [0.0, 40.0, 0.1]
    max_orders = ListParameter(default=[1, 2, 3, 4, 5, 6, 7], item_type=int)
    point_sparsity = FigureHitAndMissData.point_sparsity


    label = 'hit-and-miss'
    caption = 'Some Caption'
    standalone = True
    number_of_columns = 2

    number_of_ssa_simulations = IntParameter(default=5000)

    solver = FigureHitAndMissData.solver
    solver_kwargs = FigureHitAndMissData.solver_kwargs
    closure = FigureHitAndMiss.closure
    no_multivariate = BooleanParameter(default=False)

    def requires(self):
        hit_and_misses = [FigureHitAndMiss(max_order=max_order, timepoints_arange=self.timepoints_arange,
                                           number_of_ssa_simulations=self.number_of_ssa_simulations,
                                           point_sparsity=self.point_sparsity,
                                           solver=self.solver,
                                           solver_kwargs=self.solver_kwargs,
                                           closure=self.closure,
                                           multivariate=not self.no_multivariate)
                          for max_order in self.max_orders]

        return hit_and_misses

class FigureHitAndMissInterestingCases(TexFigureTask):

    solver = FigureSSAvMEATrajectory.solver
    solver_kwargs = FigureSSAvMEATrajectory.solver_kwargs

    number_of_ssa_simulations = FigureHitAndMissTex.number_of_ssa_simulations
    max_orders = FigureHitAndMissTex.max_orders

    label = 'hit-and-miss-interesting-cases'
    caption = ''

    closure = FigureHitAndMiss.closure
    multivariate = FigureHitAndMiss.multivariate

    parameter_set = Parameter()

    standalone = True

    number_of_columns = 2

    def requires(self):

        model = P53Model()

        requirements = []
        parameter_set = self.parameter_set

        parameters = parameter_set.parameters
        initial_conditions = parameter_set.initial_conditions
        timepoints_arange = parameter_set.timepoints_arange

        for max_order in self.max_orders:
            requirements.append(FigureSSAvMEATrajectory(timepoints_arange=timepoints_arange,
                                                        parameters=parameters,
                                                        initial_conditions=initial_conditions,
                                                        max_order=max_order,
                                                        solver=self.solver,
                                                        model=model,
                                                        solver_kwargs=self.solver_kwargs,
                                                        number_of_ssa_simulations=self.number_of_ssa_simulations,
                                                        title='Point at {0} ({1})'.format(parameter_set.label,
                                                                                          parameter_set.marker),
                                                        closure=self.closure,
                                                        multivariate=self.multivariate))

        return requirements

class HitAndMissAll(Task):
    interesting_kwargs = [dict(solver='ode15s', max_orders=range(1, 8)),
                          dict(solver='dopri5'),
                          dict(solver='rodas', max_orders=range(1, 7), point_sparsity=0.2),
                          dict(solver='euler', solver_kwargs=[('h', 0.01)])
                          ]
    def requires(self):
        tasks = []

        for kwargs in self.interesting_kwargs:

            tasks.append(FigureHitAndMissTex(**kwargs))

            for parameter_set in INTERESTING_PARAMETER_SETS:
                if 'star' in parameter_set.marker:
                    continue # We know a lot about that one already

                kwargs_to_keep = {'solver', 'solver_kwargs', 'number_of_ssa_simulations', 'max_orders',
                                  'closure', 'multiariate'}

                interesting_parameters_kwargs = {key: value for key, value in kwargs.iteritems()
                                                 if key in kwargs_to_keep}

                tasks.append(FigureHitAndMissInterestingCases(parameter_set=parameter_set,
                                                              **interesting_parameters_kwargs))

        return tasks

    def _return_object(self):
        raise Exception('This task intentionally raises exception, '
                        'as it is main goal is to just try out requires()')

if __name__ == '__main__':
    run(main_task_cls=HitAndMissAll)