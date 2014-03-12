import sys
sys.path.append("../utils")
from report_unit import ReportUnit
import pylab as pl
import pickle
import math
# this rejects unstable trajectories
REJECTED_TRAJ_THRESHOLD = 10e5
FILE_NAME = "../data_closure_and_max_order_p53.pickle"
class MyFigureA(ReportUnit):
    def __init__(self):
        super(MyFigureA, self).__init__()

    def run(self):
        with open(FILE_NAME) as f:
            list_of_dict = pickle.load(f)

        n_species = len(list_of_dict[0]["trajectories"])

        max_order = max([d["max_order"] for d in list_of_dict if d["method"] == "MEA"])


        ref_trajs = [d["trajectories"] for d in list_of_dict if  d["method"] == "SSA"][0]
        ranges_ref_trajs = [(min(r.values), max(r.values)) for r in  ref_trajs]
        ylims = [((mi - (ma - mi) /2.0), ma + (ma - mi) /2.0) for mi,ma in ranges_ref_trajs]
        print ylims



        f, axarr = pl.subplots(max_order-1, n_species, sharex=True, figsize=(9.0, 16.0))



        for mo in range(2,max_order+1):
            for sps, yl in zip(range (n_species), ylims):
                print (sps,mo)
                for d in list_of_dict:
                    if  d["method"] == "SSA":
                        y =d["trajectories"][sps].values
                        x =d["trajectories"][sps].timepoints
                        color = "k"
                        lwd = 3
                        style="-"
                        alpha=1

                    elif d["max_order"] == mo and d["trajectories"]:

                        y =d["trajectories"][sps].values
                        x =d["trajectories"][sps].timepoints

                        if d["closure"] == "scalar":
                            color="b"
                        elif d["closure"] == "log-normal":
                            color="r"
                        elif d["closure"] == "normal":
                            color="m"
                        else:
                            raise Exception("unexpected closure: {0}".format(d["closure"]))

                        if not "multivariate" in d.keys():
                            style = "-"
                            lwd=1.5
                            alpha=0.5

                        elif d["multivariate"]:
                            style = "-"
                            lwd=1.5
                            alpha=0.3
                        else:
                            style = "--"
                            lwd=1.5
                            alpha=1

                    else:
                        continue

                    ax =axarr[mo-2, sps]

                    ax.plot(x,y,linestyle=style, color=color, linewidth=lwd,alpha=alpha)
                    text="species: y_{0}; max order: {1}".format(sps,mo)

                    ax.text(0.95, 0.95, text,
                        verticalalignment='top', horizontalalignment='right', fontsize=8, transform=ax.transAxes)
                    if sps ==0:
                        ax.set_ylabel('#Molecules')
                    ax.set_xlabel('time')
                    ax.axis( ymin=yl[0], ymax=yl[1])
                    ax.tick_params(top='off', bottom='on', left='on', right='off')
                    f.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0., wspace=.1)

        pl.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)

        pl.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

        pl.suptitle("Effect of max order an closure method on\nmoment expansion approximation", fontsize=14)
        pl.savefig('figureA.pdf')
        pl.close()

        self.out_object = None

class MyFigureB(ReportUnit):
    def __init__(self):
        super(MyFigureB, self).__init__()

    def calc_distance(self, trajectories):

        diffs =[(s.values - t.values) ** 2 for s, t  in zip (self.ssa_reference, trajectories)]
        return math.log10(sum([sum(d) for d in diffs])/1000)

    def run(self):
        with open(FILE_NAME) as f:
            list_of_dict = pickle.load(f)

        self.ssa_reference = [d["trajectories"] for d in list_of_dict if d["method"] == "SSA"][0]

        closure = [
               {"closure":"scalar"},
               {"closure":"log-normal", "multivariate":True},
               {"closure":"log-normal", "multivariate":False},
               {"closure":"normal", "multivariate":True},
               {"closure":"normal", "multivariate":False}
               ]
        #
        #

        for d in list_of_dict:
            if d["method"] == "SSA":
                continue
            else:
                d["distance_to_ref"] = self.calc_distance(d["trajectories"])


        for clo_arg in closure:
            clo_arg["x_list"] = []
            clo_arg["y_list"] = []
            for d in list_of_dict:
                if d["method"] == "SSA":
                    continue

                if d["closure"] == clo_arg["closure"] and d["distance_to_ref"]:
                    if "multivariate" in d.keys():
                        if d["multivariate"] == clo_arg["multivariate"]:
                            clo_arg["x_list"].append(d["max_order"] )
                            clo_arg["y_list"].append(d["distance_to_ref"] )
                    else:
                        clo_arg["x_list"].append(d["max_order"] )
                        clo_arg["y_list"].append(d["distance_to_ref"] )


        pl.figure(figsize=(16.0, 9.0))
        pl.ylabel('log10(Distance to GSSA) (a.u.)')
        pl.xlabel('Max order')
        for clo_arg in closure:
            if clo_arg["closure"] == "scalar":
                color="b"
            elif clo_arg["closure"] == "log-normal":
                color="r"
            elif clo_arg["closure"] == "normal":
                color="m"
            else:
                raise Exception("unexpected closure: {0}".format(clo_arg["closure"]))
            lab = "closure: " + clo_arg["closure"]
            if not "multivariate" in clo_arg.keys():
                style = "-"
                lwd=2
                alpha= 0.5

            elif clo_arg["multivariate"]:
                lab = lab +"; multivariate"
                style = "-"
                alpha=0.3
                lwd=2
            else:
                lab = lab +"; univariate"
                style = "--"
                alpha=1
                lwd=2

            pl.plot(clo_arg["x_list"], clo_arg["y_list"],color=color, linewidth=lwd, linestyle=style, marker='o', label=lab, alpha=alpha)
            pl.legend()



        pl.savefig('figureB.pdf')
        pl.close()

        self.out_object = None


MyFigureA()
MyFigureB()
