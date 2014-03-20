from means.pipes import *
import numpy as np

class LotsOfTrajectoriesTask(Task):

    def requires(self):
        model_name = 'p53'
        max_order = 3

        parameters = []
        for c_2 in np.arange(1.5, 2, 0.1):
            parameters.append([90, 0.002, c_2, 1.1, 0.93, 0.96, 0.01])

        initial_conditions = [70, 30, 60]
        timepoints = [0, 40, 0.1]

        return [TrajectoryTask(model_name=model_name, max_order=max_order, parameters=x,
                           initial_conditions=initial_conditions, timepoints_arange=timepoints) for x in parameters]

    def _return_object(self):

        inputs = self.input()

        return [i.load() for i in inputs]

if __name__ == '__main__':
    run()