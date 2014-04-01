from means.pipes import *
from means.examples.sample_models import MODEL_P53

class FigurePlotTrajectExple(FigureTask):

    def requires(self):
        constants = [90, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01]
        initial_conditions = [70, 30, 60]
        time = [0, 100, 0.1]


        out = TrajectoryTask(model=MODEL_P53, max_order=2,
                   parameters=constants,
                   initial_conditions=initial_conditions, timepoints_arange=time,
                   closure="scalar", multivariate=True, do_preload=False)
        return out

    def _return_object(self):
        import pylab as pl

        traj = self.input().load()
        fig = pl.figure(figsize=(16.0, 9.0))

        return traj._create_figure()


if __name__ == '__main__':
    run(main_task_cls=FigurePlotTrajectExple)