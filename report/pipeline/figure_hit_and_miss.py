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
from means import TrajectoryCollection, SolverException
from means.pipes import *
import numpy as np
import itertools

from means.util.logs import get_logger
logger = get_logger(__name__)


class P53Model(means.Model):
    """
    A wrapper around means. Model that initialises it with P53 parameters and changes the __unicode__ function to
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

class FigureHitAndMissData(Task):

    max_order = TrajectoryTask.max_order
    timepoints_arange = TrajectoryTask.timepoints_arange
    number_of_ssa_simulations = SSATrajectoryTask.n_simulations
    point_sparsity = FloatParameter(default=0.1)

    solver = TrajectoryTask.solver
    solver_kwargs = TrajectoryTask.solver_kwargs

    def requires(self):
        model = P53Model()
        max_order = self.max_order

        initial_conditions = [70.0, 30.0, 60.0]

        parameters = []
        for c_2 in np.arange(1.5, 2.7, self.point_sparsity):
            for c_4 in np.arange(0.6, 2.5, self.point_sparsity):
                parameters.append([90.0, 0.002, round(c_2, 6), 1.1, round(c_4, 6), 0.96, 0.01])

        # We want to specify all the trajectories we need to compute as requirements of this task,
        # so luigi handles their execution and scheduling, not us.
        regular_trajectories = [TrajectoryTask(model=model, max_order=max_order,
                                parameters=x,
                                initial_conditions=initial_conditions, timepoints_arange=self.timepoints_arange,
                                solver=self.solver,
                                solver_kwargs=self.solver_kwargs)
                                for x in parameters]

        ssa_trajectories = [SSATrajectoryTask(model=model,
                                              parameters=x,
                                              initial_conditions=initial_conditions,
                                              timepoints_arange=self.timepoints_arange,
                                              n_simulations=self.number_of_ssa_simulations)
                                              for x in parameters]

        return regular_trajectories + ssa_trajectories

    def _distance_between_trajectories(self, lookup, trajectory_collection):
        distance = 0.0
        for trajectory_a in trajectory_collection:
            try:
                trajectory_b = lookup[trajectory_a.description]
            except KeyError:
                continue

            distance += np.sum(np.square(trajectory_a.values - trajectory_b.values))

        return distance

    def _return_object(self):
        success_x = []
        success_y = []
        distances = []

        failure_x = []
        failure_y = []
        failure_c = []


        ssa_trajectory_lookup = {}

        for task, trajectory_buffer in itertools.izip(self.requires(), self.input()):
            if isinstance(task, SSATrajectoryTask):
                lookup = {}
                for trajectory in trajectory_buffer.load():
                    lookup[trajectory.description] = trajectory

                ssa_trajectory_lookup[tuple(task.parameters)] = lookup

        for task, trajectory_buffer in itertools.izip(self.requires(), self.input()):
            if isinstance(task, SSATrajectoryTask):
                continue

            x, y = task.parameters[2], task.parameters[4]
            trajectory = trajectory_buffer.load()
            if isinstance(trajectory, SolverException):
                failure_x.append(x)
                failure_y.append(y)
                base_exception = trajectory.base_exception
                try:
                    failure_time = base_exception.t
                except AttributeError:
                    # Some solvers, i.e. RODAS do not give the time at failure.
                    failure_time = np.nan
                failure_c.append(failure_time)
            elif isinstance(trajectory, TrajectoryCollection):
                success_x.append(x)
                success_y.append(y)
                ssa_equivalent = ssa_trajectory_lookup[tuple(task.parameters)]
                distance = self._distance_between_trajectories(ssa_equivalent, trajectory)
                distances.append(distance)
            else:
                raise Exception('Got {0!r} as trajectory, expected either SolverException'
                                ' or TrajectoryCollection'.format(trajectory))

        return success_x, success_y, distances, failure_x, failure_y, failure_c

class FigureHitAndMiss(FigureTask):

    max_order = FigureHitAndMissData.max_order
    timepoints_arange = FigureHitAndMissData.timepoints_arange
    number_of_ssa_simulations = FigureHitAndMissData.number_of_ssa_simulations
    point_sparsity = FigureHitAndMissData.point_sparsity

    # We need other max_orders to be able to compute colour ranges
    max_orders = ListParameter(item_type=int)

    solver = FigureHitAndMissData.solver
    solver_kwargs = FigureHitAndMissData.solver_kwargs

    def requires(self):

        # Let's make sure self.max_order is always first
        requirements = [FigureHitAndMissData(max_order=self.max_order, timepoints_arange=self.timepoints_arange,
                                             number_of_ssa_simulations=self.number_of_ssa_simulations,
                                             point_sparsity=self.point_sparsity,
                                             solver=self.solver,
                                             solver_kwargs=self.solver_kwargs)]

        # Add all other orders
        for order in self.max_orders:
            if order != self.max_order:
                requirements.append(FigureHitAndMissData(max_order=order, timepoints_arange=self.timepoints_arange,
                                             number_of_ssa_simulations=self.number_of_ssa_simulations,
                                             point_sparsity=self.point_sparsity,
                                             solver=self.solver,
                                             solver_kwargs=self.solver_kwargs))
        return requirements

    def _return_object(self):
        from matplotlib import pyplot as plt
        from matplotlib import colors
        fig = plt.figure()

        all_data = self.input()

        min_dist = float('inf')

        all_input_distances = []
        for input_ in all_data:
            __, __,  input_distances, __, __, __ = input_.load()
            all_input_distances.extend(input_distances)

        # Get 99.5th percentile as the max_dist (i.e. exclude some outliers)
        max_dist = np.percentile(all_input_distances, 99.5)

        # it is less-confusing if we use something close to zero for min_dist
        min_dist = 1e-2

        success_x, success_y, success_c, failure_x, failure_y, failure_c = self.input()[0].load()

        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlabel('c_2')
        ax.set_ylabel('c_4')
        ax.set_title('max_order = {0}'.format(self.max_order))

        # Sky Blue -> Blue
        cdict_success = {'red': ((0.0, 0.0, 0.0),
                                 (1.0, 0.0, 0.0)),

                         'green': ((0.0, 1.0, 1.0),
                                  (1.0, 0.0, 0.0)),

                         'blue': ((0.0, 0.0, 0.0),
                                 (1.0, 1.0, 1.0))
                        }
        cmap_success = colors.LinearSegmentedColormap('SkyBlueBlue', cdict_success)


        #success_scatter = ax.scatter(success_x, success_y, c=success_c, s=80, label='Success', cmap=cmap_success,
        #                             edgecolor='', norm=colors.LogNorm(), marker='s', vmin=min_dist, vmax=max_dist)
        from means.inference.plotting import plot_contour
        plot_contour(success_x, success_y, success_c, 'a', 'b', vmin=min_dist, vmax=max_dist)

        # Red -> Blue
        cdict_failure = {'red': ((0.0, 1.0, 1.0),
                                 (1.0, 0.0, 0.0)),

                         'green': ((0.0, 0.0, 0.0),
                                   (1.0, 0.0, 0.0)),

                         'blue': ((0.0, 0.0, 0.0),
                                  (1.0, 1.0, 1.0))
                          }

        cmap_failure = colors.LinearSegmentedColormap('RedBlue', cdict_failure)


        vmin = self.timepoints_arange[0]
        vmax = self.timepoints_arange[1]

        failure_x = np.array(failure_x)
        failure_y = np.array(failure_y)
        failure_c = np.array(failure_c)

        nan_failure_times = np.isnan(failure_c)
        failure_without_time_x, failure_without_time_y = failure_x[nan_failure_times], failure_y[nan_failure_times]

        failure_x = failure_x[~nan_failure_times]
        failure_y = failure_y[~nan_failure_times]
        failure_c = failure_c[~nan_failure_times]

        if len(failure_x):
            failure_scatter = ax.scatter(failure_x, failure_y, marker='^', s=80, c=failure_c, cmap=cmap_failure,
                                         vmin=vmin, vmax=vmax, label='Failure', edgecolor='')
        else:
            failure_scatter = None

        if len(failure_without_time_x):
            failure_no_time_scatter = ax.scatter(failure_without_time_x, failure_without_time_y,
                                                 marker='^', s=80, c='r', edgecolor='', label='Failure')

        #failure_circles_scatter = ax.scatter(failure_x, failure_y, marker='s', s=80, facecolors='', edgecolors='r')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        # if len(success_x):
        #     colorbar_success = plt.colorbar(success_scatter, ax=ax)
        #     colorbar_success.set_label('Sum-of-squares distance from SSA result')

        if failure_scatter:
            colorbar_failure = plt.colorbar(failure_scatter, ax=ax)
            colorbar_failure.set_label('Point of failure')

        return fig

class FigureSSAvMEATrajectory(FigureTask):

    model = TrajectoryTask.model

    timepoints_arange = TrajectoryTask.timepoints_arange
    parameters = TrajectoryTask.parameters
    initial_conditions = TrajectoryTask.initial_conditions

    max_order = TrajectoryTask.max_order

    solver = TrajectoryTask.solver
    solver_kwargs = TrajectoryTask.solver_kwargs

    number_of_ssa_simulations = SSATrajectoryTask.n_simulations

    def requires(self):

        ssa_trajectory = SSATrajectoryTask(model=self.model,
                                           parameters=self.parameters,
                                           initial_conditions=self.initial_conditions,
                                           timepoints_arange=self.timepoints_arange,
                                           n_simulations=self.number_of_ssa_simulations)

        trajectory = TrajectoryTask(model=self.model, max_order=self.max_order,
                                    parameters=self.parameters,
                                    initial_conditions=self.initial_conditions,
                                    timepoints_arange=self.timepoints_arange,
                                    solver=self.solver,
                                    solver_kwargs=self.solver_kwargs)

        return [ssa_trajectory, trajectory]

    def _return_object(self):
        import matplotlib.pyplot as plt
        figure = plt.figure()

        ssa_trajectory_buffer, trajectory_buffer = self.input()
        ssa_trajectories = ssa_trajectory_buffer.load()
        ssa_trajectories = [ssa_trajectories[0]] # Let's take only the first trajectory
        trajectories = trajectory_buffer.load()

        number_of_ssa_trajectories = len(ssa_trajectories)

        for i, (ssa_trajectory, trajectory) in enumerate(zip(ssa_trajectories, trajectories)):
            assert(ssa_trajectory.description == trajectory.description)

            plt.subplot(2, number_of_ssa_trajectories, i+1)
            plt.title(ssa_trajectory.description.mathtext())

            ssa_trajectory.plot(label='SSA')
            trajectory.plot(label='Solver')
            plt.legend()

            plt.subplot(2, number_of_ssa_trajectories, i+1+number_of_ssa_trajectories)
            plt.title(ssa_trajectory.description.mathtext() + ' - difference')
            (ssa_trajectory - trajectory).plot(label='difference')


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

    def requires(self):
        hit_and_misses = [FigureHitAndMiss(max_order=max_order, timepoints_arange=self.timepoints_arange,
                                 number_of_ssa_simulations=self.number_of_ssa_simulations,
                                 point_sparsity=self.point_sparsity,
                                 max_orders=self.max_orders,
                                 solver=self.solver,
                                 solver_kwargs=self.solver_kwargs)
                for max_order in self.max_orders]

        return hit_and_misses

class FigureHitAndMissInterestingCases(TexFigureTask):

    model = P53Model()

    solver = FigureSSAvMEATrajectory.solver
    solver_kwargs = FigureSSAvMEATrajectory.solver_kwargs

    number_of_ssa_simulations = FigureHitAndMissTex.number_of_ssa_simulations

    label = 'hit-and-miss-interesting-cases'
    caption = ''


    def requires(self):

        parameters = [90.0, 0.002, 2.5, 1.1, 1.8, 0.96, 0.01]
        initial_conditions = [70.0, 30.0, 60.0]
        timepoints_arange = FigureHitAndMissTex.timepoints_arange
        max_order = 1

        return [FigureSSAvMEATrajectory(model=self.model,
                                        timepoints_arange=timepoints_arange,
                                        parameters=parameters,
                                        initial_conditions=initial_conditions,
                                        max_order=max_order,
                                        solver=self.solver,
                                        solver_kwargs=self.solver_kwargs,
                                        number_of_ssa_simulations=self.number_of_ssa_simulations)]



if __name__ == '__main__':
    #run(main_task_cls=FigureHitAndMissTex)
    run()