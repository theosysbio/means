import luigi
from means.pipes import *
import itertools


class FigureClosureAndMaxOrderData(Task):


    timepoints_arange = ListParameter()
    initial_conditions = ListParameter()
    simul_params = ListParameter()
    model_name = luigi.Parameter()
    max_max_order = IntParameter()
    n_simulations = IntParameter(default=20)

    def requires(self):


        regular_trajectories = [TrajectoryTask(model_name=self.model_name, max_order=mo,
                               parameters=self.simul_params,
                               initial_conditions=self.initial_conditions, timepoints_arange=self.timepoints_arange)
                               for mo in range(2, self.max_max_order + 1)]

        ssa_trajectories = [SSATrajectoryTask(model_name=self.model_name,
                               parameters=self.simul_params,
                               initial_conditions=self.initial_conditions, timepoints_arange=self.timepoints_arange,
                               n_simulations=self.n_simulations)]

        return regular_trajectories + ssa_trajectories

    def _return_object(self):
        out = []
        for task, trajectory_buffer in itertools.izip(self.requires(), self.input()):
            out.append((task, trajectory_buffer.load()))
        return out




class FigureP53Data(FigureClosureAndMaxOrderData):

    max_max_order = IntParameter(default=3)

    timepoints_arange = [0, 40, 0.1]
    initial_conditions = [70, 30, 60]
    simul_params = [90, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01]
    model_name = "p53"

    def requires(self):
        out = FigureClosureAndMaxOrderData( simul_params=self.simul_params,
                                            initial_conditions=self.initial_conditions,
                                            timepoints_arange=self.timepoints_arange,
                                            max_max_order=self.max_max_order,
                                            model_name=self.model_name)
        return out

    def _return_object(self):
        return self.input().load()

class FigureP53(FigureTask):

    dummy=IntParameter(default=1)
    def requires(self):
        out = FigureP53Data()
        return out

    def _return_object(self):
        from matplotlib import pyplot as plt

        fig = plt.figure()

        input_ = self.input()
        all_trajs = input_.load()
        for task, trajectory_buffer in all_trajs:
            if isinstance(task, SSATrajectoryTask):
                trajectory_buffer.plot()
                continue

        return fig

if __name__ == '__main__':
    run(main_task_cls=FigureP53)