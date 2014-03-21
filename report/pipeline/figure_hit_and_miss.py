from means import TrajectoryCollection, SolverException
from means.pipes import *
import numpy as np
import itertools
from means.util.logs import get_logger


logger = get_logger(__name__)

class FigureHitAndMissData(Task):

    max_order = IntParameter()
    timepoints_arange = ListParameter()

    def requires(self):
        model_name = 'p53'
        max_order = self.max_order

        initial_conditions = [70, 30, 60]

        parameters = []
        for c_2 in np.arange(1.5, 2.5, 0.1):
            for c_4 in np.arange(0.8, 2.5, 0.1):
                parameters.append([90, 0.002, c_2, 1.1, c_4, 0.96, 0.01])


        return [TrajectoryTask(model_name=model_name, max_order=max_order,
                               parameters=x,
                               initial_conditions=initial_conditions, timepoints_arange=self.timepoints_arange)
                for x in parameters]

    def _return_object(self):
        success_x = []
        success_y = []
        failure_x = []
        failure_y = []
        failure_c = []

        for task, trajectory_buffer in itertools.izip(self.requires(), self.input()):
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

    max_order = IntParameter()
    timepoints_arange = ListParameter()

    def requires(self):
        return FigureHitAndMissData(max_order=self.max_order, timepoints_arange=self.timepoints_arange)

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

    # Note that this is not a parameter, it is a constant
    timepoints_arange = [0, 40, 0.1]
    max_orders = [1, 2, 3, 4, 5]

    def requires(self):
        return [FigureHitAndMiss(max_order=max_order, timepoints_arange=self.timepoints_arange)
                for max_order in self.max_orders]

class FigureHitAndMissMain(Task):
    """
    Convenience function to be able to run file without command line args
    """

    def requires(self):
        return FigureHitAndMissTex(label='hit-and-miss', caption='Some caption',
                                   standalone=True)

    def _return_object(self):
        return None

if __name__ == '__main__':
    run(main_task_cls=FigureHitAndMissMain)