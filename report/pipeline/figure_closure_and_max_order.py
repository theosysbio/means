import luigi
from means.pipes import *
from means.inference.distances import sum_of_squares
import itertools


N_SSA = 5000

CLOSURE_DICTS = [
        {"closure": "scalar",        "multivariate": True, "col": "b", "sty": "-", "alpha": 0.3},
        {"closure": "normal",        "multivariate": True, "col": "m", "sty": "-", "alpha": 0.3},
        {"closure": "normal",        "multivariate": False, "col": "m", "sty": "--", "alpha": 0.5},
        {"closure": "log-normal",    "multivariate": True, "col": "r", "sty": "-", "alpha": 0.3},
        {"closure": "log-normal",    "multivariate": False, "col": "r", "sty": "--", "alpha": 0.5}
        ]

def get_traject(tasks, trajectory_buffers, max_order, sp, closure, multiv, ssa=False):
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

class P53Model(means.Model):
    def __init__(self):
        from means.examples import MODEL_P53
        super(P53Model, self).__init__(MODEL_P53.species, MODEL_P53.parameters, MODEL_P53.propensities,
                                       MODEL_P53.stoichiometry_matrix)
    def __str__(self):
        return 'p53'

class DataClosureAndMaxOrder(Task):


    timepoints_arange = ListParameter()
    initial_conditions = ListParameter()
    simul_params = ListParameter()
    model = ModelParameter()
    max_max_order = IntParameter()
    n_simulations = IntParameter(default=N_SSA)

    def requires(self):
        regular_trajectories = []

        for clos_multiv in CLOSURE_DICTS:

            regular_trajectories.extend([TrajectoryTask(model=self.model, max_order=mo,
                               parameters=self.simul_params,
                               initial_conditions=self.initial_conditions, timepoints_arange=self.timepoints_arange,
                               closure=clos_multiv["closure"], multivariate=clos_multiv["multivariate"], do_preload=False)

                               for mo in range(2, self.max_max_order + 1)])

        ssa_trajectories = [SSATrajectoryTask(model=self.model,
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

                ssa_traj = get_traject(tasks, trajectory_buffers, 0, sps,
                                            0, 0, ssa=True)
                mi, ma = min(ssa_traj.values), max(ssa_traj.values)
                yl = ((mi - (ma - mi) /2.0), (ma + (ma - mi) /2.0))

                ax.plot(ssa_traj.timepoints, ssa_traj.values, color="k", linewidth=3)

                for clos_multiv in CLOSURE_DICTS:

                    traj = get_traject(tasks, trajectory_buffers, max_order=mo, sp=sps,
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

class FigureSummaryDistanceBase(FigureTask):
    def _return_object(self):



        input_ = self.input()
        tasks, trajectory_buffers = zip(*input_.load())


        results = CLOSURE_DICTS[:]

        for trajs in trajectory_buffers:
            if not isinstance(trajs, Exception):
                n_species = len([None for t in trajs if t.description.order == 1])
                continue


        ssa_trajs = [get_traject(tasks, trajectory_buffers, 0, sps,
                                            0, 0, ssa=True)
                     for sps in range(n_species)]

        ssa_dict={}
        for s in ssa_trajs:
            ssa_dict[s.description] = s

        for clos_multiv in results:
            clos_multiv["distances"] = []
            clos_multiv["max_orders"] = []

            for mo in range(2, self.max_max_order+1):
                trajs = [get_traject(tasks, trajectory_buffers, max_order=mo, sp=sps,
                                            closure=clos_multiv["closure"],
                                            multiv=clos_multiv["multivariate"])
                                    for sps in range(n_species)]

                if len([True for t in trajs if not isinstance(t, Exception)]) == n_species:
                   clos_multiv["distances"].append(sum_of_squares(trajs, ssa_dict))
                   clos_multiv["max_orders"].append(mo)


        import pylab as pl
        fig = pl.figure(figsize=(16.0, 6.0))
        pl.yscale('log')
        #fig.set_yscale('log')
        for res in results:
            if res["multivariate"]:
                mv = "multivariate"
            else:
                mv = "univariate"
            if res["closure"] == "scalar":
                mv = "value=0"

            lab = "{0}, {1}".format(res["closure"], mv)
            pl.plot(res["max_orders"], res["distances"], color=res["col"],
                    linestyle=res["sty"], alpha=res["alpha"], marker="o", label=lab)

        pl.axis(xmin=1.5, xmax=self.max_max_order + 0.5)

        pl.legend(title="Closure", loc=6)
        pl.ylabel('Sum of square Distance to GSSA')

        pl.xlabel('Max order')
        return fig

class DataP53(DataClosureAndMaxOrder):
    timepoints_arange = [0, 40, 0.1]
    initial_conditions = [70, 30, 60]
    simul_params = [90, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01]
    model = P53Model()

class FigureP53(FigureClosureMaxOrderBase):
    max_max_order = IntParameter(default=7)
    def requires(self):
        out = DataP53(max_max_order = self.max_max_order)
        return out

class FigureP53Summary(FigureSummaryDistanceBase):
    max_max_order = IntParameter(default=7)
    def requires(self):
        out = DataP53(max_max_order = self.max_max_order)
        return out


class FigureP53Simple(FigureP53):

    def _return_object(self):
        import pylab as pl

        input_ = self.input()
        tasks, trajectory_buffers = zip(*input_.load())

        for trajs in trajectory_buffers:
            if not isinstance(trajs, Exception):
                n_species = len([None for t in trajs if t.description.order == 1])
                continue

        # f, axarr = pl.subplots(1, 2, sharex=True, figsize=(9.0, 9.0))
        f, axarr = pl.subplots(1, 2, sharey=True, figsize=(16.0, 6.0))
        sps= 0
        ssa_traj = get_traject(tasks, trajectory_buffers, 0, sps,
                                            0, 0, ssa=True)
        mi, ma = min(ssa_traj.values), max(ssa_traj.values)
        yl = ((mi - (ma - mi) /5.0), (ma + (ma - mi) /10.0))


        for i,mo in enumerate([3,7]):
            ax = axarr[i]
            for clos_multiv in CLOSURE_DICTS:
                traj = get_traject(tasks, trajectory_buffers, max_order=mo, sp=sps,
                                            closure=clos_multiv["closure"],
                                            multiv=clos_multiv["multivariate"])

                if isinstance(traj, Exception):
                    continue
                ax.plot(ssa_traj.timepoints, ssa_traj.values, color="k", linewidth=2)
                ax.plot(traj.timepoints, traj.values, color=clos_multiv["col"],
                            linestyle=clos_multiv["sty"],
                            alpha=clos_multiv["alpha"]
                    )

                f.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0., wspace=0)
                text="Max order: {0}".format(mo)
                ax.text(0.95, 0.95, text,
                    verticalalignment='top', horizontalalignment='right', fontsize=11, transform=ax.transAxes)
                if i ==0:
                    ax.set_ylabel('#Molecules')
                ax.set_xlabel('time')
                ax.axis(ymin=yl[0], ymax=yl[1])

        pl.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
        return f


class AllFigures(Task):

    def requires(self):
        out = []
        out += [FigureP53, FigureP53Summary]

        return out

    def _return_object(self):
        return Task()

# class FigureHes1Data(FigureClosureMaxOrderDataBase):
#     timepoints_arange = [0, 240, 1]
#     initial_conditions = [4, 20, 20]
#     simul_params = [5, 10, 1, 1]
#     #model = Hes1Model()
#
# class FigureHes1(FigureClosureMaxOrderBase):
#     max_max_order = IntParameter(default=3)
#     def requires(self):
#         out = FigureHes1Data(max_max_order=self.max_max_order)
#         return out

if __name__ == '__main__':
    # run(main_task_cls=FigureHes1)
    run(main_task_cls=FigureP53Summary)
    run(main_task_cls=FigureP53Simple)
    # run(main_task_cls=FigureP53Summary)