import luigi
from means.pipes import *
import itertools


CLOSURE_DICTS = [
        {"closure": "scalar",        "multivariate": True, "col": "b", "sty": "-", "alpha": 0.3},
        {"closure": "normal",        "multivariate": True, "col": "m", "sty": "-", "alpha": 0.3},
        {"closure": "normal",        "multivariate": False, "col": "m", "sty": "--", "alpha": 0.5},
        {"closure": "log-normal",    "multivariate": True, "col": "r", "sty": "-", "alpha": 0.3},
        {"closure": "log-normal",    "multivariate": False, "col": "r", "sty": "--", "alpha": 0.5}
        ]

class FigureClosureAndMaxOrderData(Task):


    timepoints_arange = ListParameter()
    initial_conditions = ListParameter()
    simul_params = ListParameter()
    model_name = luigi.Parameter()
    max_max_order = IntParameter()
    n_simulations = IntParameter(default=20)

    def requires(self):
        regular_trajectories = []

        for clos_multiv in CLOSURE_DICTS:

            regular_trajectories.extend([TrajectoryTask(model_name=self.model_name, max_order=mo,
                               parameters=self.simul_params,
                               initial_conditions=self.initial_conditions, timepoints_arange=self.timepoints_arange,
                               closure=clos_multiv["closure"], multivariate=clos_multiv["multivariate"])

                               for mo in range(2, self.max_max_order + 1)])

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

class FigureClosureMaxOrderBase(FigureTask):

    def get_traject(self, tasks, trajectory_buffers, max_order, sp, closure, multiv, ssa=False):
        for ta, trajs in itertools.izip(tasks, trajectory_buffers):

            if isinstance(ta, SSATrajectoryTask):
                if ssa:
                    for t in trajs:
                        if t.description.order == 1 and t.description.n_vector[sp]:
                            return t
                continue
            elif ta.closure == closure and \
                    ta.max_order == max_order and \
                    ta.multivariate == multiv:
                if isinstance(trajs, Exception):
                    return trajs

                for t in trajs:
                    if t.description.order == 1 and t.description.n_vector[sp]:
                        return t

        raise Exception("Trajectory not found!!")


    def _return_object(self):
        import pylab as pl

        input_ = self.input()
        tasks, trajectory_buffers = zip(*input_.load())

        for trajs in trajectory_buffers:
            if not isinstance(trajs, Exception):
                n_species = len([None for t in trajs if t.description.order == 1])
                continue

        f, axarr = pl.subplots(self.max_max_order - 1, n_species, sharex=True, figsize=(9.0, 16.0))

        for mo in range(2, self.max_max_order+1):
            for sps in range(n_species):
                ax = axarr[mo-2, sps]

                ssa_traj = self.get_traject(tasks, trajectory_buffers, 0, sps,
                                            0, 0, ssa=True)
                mi, ma = min(ssa_traj.values), max(ssa_traj.values)
                yl = ((mi - (ma - mi) /2.0), (ma + (ma - mi) /2.0))

                ax.plot(ssa_traj.timepoints, ssa_traj.values, color="k", linewidth=3)

                for clos_multiv in CLOSURE_DICTS:

                    traj = self.get_traject(tasks, trajectory_buffers, max_order=mo, sp=sps,
                                            closure=clos_multiv["closure"],
                                            multiv=clos_multiv["multivariate"])


                    if isinstance(traj, Exception):
                        continue

                    ax.plot(traj.timepoints, traj.values, color=clos_multiv["col"],
                            linestyle=clos_multiv["sty"],
                            alpha=clos_multiv["alpha"]
                    )

                    text="species: y_{0}; max order: {1}".format(sps, mo)

                    ax.text(0.95, 0.95, text,
                        verticalalignment='top', horizontalalignment='right', fontsize=8, transform=ax.transAxes)
                    if sps ==0:
                        ax.set_ylabel('#Molecules')
                    ax.set_xlabel('time')
                    ax.axis(ymin=yl[0], ymax=yl[1])
                    ax.tick_params(top='off', bottom='on', left='on', right='off')
                    f.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0., wspace=.1)

        pl.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
        pl.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        pl.suptitle("Effect of max order an closure method on\nmoment expansion approximation", fontsize=14)

        return f

class FigureClosureMaxOrderDataBase(FigureClosureAndMaxOrderData):

    max_max_order = IntParameter()

    def requires(self):
        out = FigureClosureAndMaxOrderData( simul_params=self.simul_params,
                                            initial_conditions=self.initial_conditions,
                                            timepoints_arange=self.timepoints_arange,
                                            max_max_order=self.max_max_order,
                                            model_name=self.model_name)
        return out
    def _return_object(self):
        return self.input().load()

class FigureP53Data(FigureClosureMaxOrderDataBase):
    timepoints_arange = [0, 40, 0.1]
    initial_conditions = [70, 30, 60]
    simul_params = [90, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01]
    model_name = "p53"

class FigureP53(FigureClosureMaxOrderBase):
    max_max_order = IntParameter(default=3)
    def requires(self):
        out = FigureP53Data(max_max_order=self.max_max_order)
        return out

class FigureHes1Data(FigureClosureMaxOrderDataBase):
    timepoints_arange = [0, 240, 1]
    initial_conditions = [4, 20, 20]
    simul_params = [5, 10, 1, 1]
    model_name = "hes1"

class FigureHes1(FigureClosureMaxOrderBase):
    max_max_order = IntParameter(default=3)
    def requires(self):
        out = FigureHes1Data(max_max_order=self.max_max_order)
        return out

if __name__ == '__main__':
    run(main_task_cls=FigureHes1)
    #run(main_task_cls=FigureP53)