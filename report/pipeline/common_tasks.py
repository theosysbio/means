from means import TrajectoryCollection, SolverException
from means.pipes import *
import numpy as np
import itertools
from means.util.logs import get_logger


logger = get_logger(__name__)

class FigureHitAndMissTask(FigureTask):

    max_order = IntParameter()
    # Note that this is not a parameter, it is a constant
    timepoints_arange = [0, 40, 0.1]

    def requires(self):
        model_name = 'p53'
        max_orer = self.max_order

        initial_conditions = [70, 30, 60]

        parameters = []
        for c_2 in np.arange(1.5, 2.5, 0.1):
            for c_4 in np.arange(0.8, 2.5, 0.1):
                parameters.append([90, 0.002, c_2, 1.1, c_4, 0.96, 0.01])


        return [TrajectoryTask(model_name=model_name, max_order=max_orer,
                               parameters=x,
                               initial_conditions=initial_conditions, timepoints_arange=self.timepoints_arange)
                for x in parameters]

    def _return_object(self):
        from matplotlib import pyplot as plt
        from matplotlib import colors
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
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ax.set_xlabel('c_2')
        ax.set_ylabel('c_4')
        ax.set_title('max_order = {0}'.format(self.max_order))
        ax.scatter(success_x, success_y, color='b', label='Success')

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
        failure_scatter = ax.scatter(failure_x, failure_y, marker='x', c=failure_c, cmap=cmap,
                                     vmin=vmin, vmax=vmax, label='Failure')

        colorbar = fig.colorbar(failure_scatter)
        colorbar.set_label('Point of failure')
        ax.legend()
        return fig

if __name__ == '__main__':
    run()