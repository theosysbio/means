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

    max_order = IntParameter()
    timepoints_arange = ListParameter()
    number_of_ssa_simulations = IntParameter()
    point_sparsity = FloatParameter(default=0.1)

    def requires(self):
        model = P53Model()
        max_order = self.max_order

        initial_conditions = [70, 30, 60]

        parameters = []
        for c_2 in np.arange(1.5, 2.5, self.point_sparsity):
            for c_4 in np.arange(0.8, 2.5, self.point_sparsity):
                parameters.append([90, 0.002, round(c_2, 6), 1.1, round(c_4, 6), 0.96, 0.01])

        # We want to specify all the trajectoreis we need to compute as requirements of this task,
        # so luigi handles their execution and scheduling, not us.
        regular_trajectories = [TrajectoryTask(model=model, max_order=max_order,
                                parameters=x,
                                initial_conditions=initial_conditions, timepoints_arange=self.timepoints_arange)
                                for x in parameters]

        ssa_trajectories = [SSATrajectoryTask(model=model,
                                              parameters=x,
                                              initial_conditions=initial_conditions,
                                              timepoints_arange=self.timepoints_arange,
                                              n_simulations=self.number_of_ssa_simulations)
                                              for x in parameters]

        return regular_trajectories + ssa_trajectories

    def _return_object(self):
        success_x = []
        success_y = []
        failure_x = []
        failure_y = []
        failure_c = []

        for task, trajectory_buffer in itertools.izip(self.requires(), self.input()):
            if isinstance(task, SSATrajectoryTask):
                continue

            x, y = task.parameters[2], task.parameters[4]
            trajectory = trajectory_buffer.load()
            if isinstance(trajectory, SolverException):
                failure_x.append(x)
                failure_y.append(y)
                failure_c.append(trajectory.base_exception.t)
            elif isinstance(trajectory, TrajectoryCollection):
                success_x.append(x)
                success_y.append(y)
            else:
                raise Exception('Got {0!r} as trajectory, expected either SolverException'
                                ' or TrajectoryCollection'.format(trajectory))

        return success_x, success_y, failure_x, failure_y, failure_c

class FigureHitAndMiss(FigureTask):

    max_order = FigureHitAndMissData.max_order
    timepoints_arange = FigureHitAndMissData.timepoints_arange
    number_of_ssa_simulations = FigureHitAndMissData.number_of_ssa_simulations
    point_sparsity = FigureHitAndMissData.point_sparsity

    def requires(self):
        # I split the data aggregation and figure plotting into different tasks, thus this dependancy
        # this is not strictly necessary, if you do not plot figures into subplots, (like we don't here)
        # But would be desired if we do, essentially this dependency can be refactored to be implicit
        return FigureHitAndMissData(max_order=self.max_order, timepoints_arange=self.timepoints_arange,
                                    number_of_ssa_simulations=self.number_of_ssa_simulations,
                                    point_sparsity=self.point_sparsity)

    def _return_object(self):
        from matplotlib import pyplot as plt
        from matplotlib import colors
        fig = plt.figure()

        input_ = self.input()

        success_x, success_y, failure_x, failure_y, failure_c = input_.load()
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlabel('c_2')
        ax.set_ylabel('c_4')
        ax.set_title('max_order = {0}'.format(self.max_order))
        success_scatter = ax.scatter(success_x, success_y, color='b', label='Success')

        cdict = {'red': ((0.0, 1.0, 1.0),
                         (1.0, 0.0, 0.0)),

                 'green': ((0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),

                 'blue': ((0.0, 0.0, 0.0),
                          (1.0, 1.0, 1.0))
                }

        cmap = colors.LinearSegmentedColormap('RedBlue', cdict)

        vmin = self.timepoints_arange[0]
        vmax = self.timepoints_arange[1]
        failure_scatter = ax.scatter(failure_x, failure_y, marker='s', c=failure_c, cmap=cmap,
                                     vmin=vmin, vmax=vmax, label='Failure', edgecolor='')

        ax.legend()
        if failure_x:
            colorbar = plt.colorbar(failure_scatter, ax=ax)
            colorbar.set_label('Point of failure')

        return fig

class FigureHitAndMissTex(TexFigureTask):
    """
    Join all hit and miss figures into one tex file
    """

    # Note that this is not a parameter, it is a constant
    timepoints_arange = [0, 40, 0.1]
    max_orders = ListParameter(default=[1, 2, 3, 4, 5])
    point_sparsity = FigureHitAndMissData.point_sparsity

    label = 'hit-and-miss'
    caption = 'Some Caption'
    standalone = True
    number_of_columns = 2

    number_of_ssa_simulations = IntParameter(default=500)

    def requires(self):
        return [FigureHitAndMiss(max_order=max_order, timepoints_arange=self.timepoints_arange,
                                 number_of_ssa_simulations=self.number_of_ssa_simulations,
                                 point_sparsity=self.point_sparsity)
                for max_order in self.max_orders]

if __name__ == '__main__':
    run(main_task_cls=FigureHitAndMissTex)