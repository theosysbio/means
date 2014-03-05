import sys
sys.path.append("../utils")
from report_unit import ReportUnit
import pylab as pl
import pickle
import math
# this rejects unstable trajectories
REJECTED_TRAJ_THRESHOLD = 10e5

class MyFigureA(ReportUnit):
    def __init__(self):
        super(MyFigureA, self).__init__()

    def run(self):
        with open("../data_best_order_mom.pickle") as f:
            list_of_dict = pickle.load(f)

        n_species = len(list_of_dict[0]["trajectories"])

        max_order = max([d["max_order"] for d in list_of_dict if d["method"] == "MEA"])



        f, axarr = pl.subplots(max_order-1, n_species, sharex=True, sharey=True, figsize=(9.0, 16.0))
        #f, axarr = pl.subplots(3, max_order)



        for mo in range(2,max_order+1):
            for sps in range (n_species):
                print (sps,mo)
                for d in list_of_dict:
                    if  d["method"] == "SSA":
                        y =d["trajectories"][sps].values
                        x =d["trajectories"][sps].timepoints
                        color = "k"
                        lwd = 3
                        style="-"


                    elif d["max_order"] == mo and d["trajectories"]:
                        y =d["trajectories"][sps].values
                        x =d["trajectories"][sps].timepoints

                        if d["closer"] == "zero":
                            color="b"
                        elif d["closer"] == "log-normal":
                            color="r"
                        elif d["closer"] == "normal":
                            color="m"
                        else:
                            raise Exception("unexpected closer: {0}".format(d["closer"]))

                        if not "multivariate" in d.keys():
                            style = "-"
                            lwd=2

                        elif d["multivariate"]:
                            style = "--"
                            lwd=2
                        else:
                            style = ":"
                            lwd=2

                    else:
                        continue

                    ax =axarr[mo-2, sps]
                    ax.plot(x,y,linestyle=style, color=color, linewidth=lwd)
                    text="species: y_{0}; max order: {1}".format(sps,mo)
                    ax.text(0.95, 0.95, text,
                        verticalalignment='top', horizontalalignment='right', fontsize=8, transform=ax.transAxes)
                    if sps ==0:
                        ax.set_ylabel('#Molecules')
                    ax.set_xlabel('time')

                    ax.tick_params(top='off', bottom='on', left='on', right='off')
                    f.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0, wspace=0)

        pl.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
        #pl.set_xticks([10, 20, 30, 40])
        #pl.set_yticks([20, 30, 40, 50, 60, 70,80])

        pl.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)




        #pl.ylabel('log_10(dt) (s)')
        #pl.xlabel('number of ODEs')

        #for d in dic:
        #    pl.plot(d["n_eq"], d["dt"], linewidth=2.5, linestyle='--', marker='o', label=d["legend"])
        #pl.legend(loc='upper right')
        #pl.show()
        #pl.legend()
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
        with open("../data_best_order_mom.pickle") as f:
            list_of_dict = pickle.load(f)

        self.ssa_reference = [d["trajectories"] for d in list_of_dict if d["method"] == "SSA"][0]

        closer_args = [
               {"closer":"zero"},
               {"closer":"log-normal", "multivariate":True},
               {"closer":"log-normal", "multivariate":False},
               {"closer":"normal", "multivariate":True},
               {"closer":"normal", "multivariate":False}
               ]
        #
        #

        for d in list_of_dict:
            if d["method"] == "SSA":
                continue
            if max(d["trajectories"][0].values) > REJECTED_TRAJ_THRESHOLD:
                d["distance_to_ref"] = None
                print  "WAS REJECTED:\n{0}\n______________________".format(d)
            else:
                d["distance_to_ref"] = self.calc_distance(d["trajectories"])


        for clo_arg in closer_args:
            clo_arg["x_list"] = []
            clo_arg["y_list"] = []
            for d in list_of_dict:
                if d["method"] == "SSA":
                    continue

                if d["closer"] == clo_arg["closer"] and d["distance_to_ref"]:
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
        for clo_arg in closer_args:
            if clo_arg["closer"] == "zero":
                color="b"
            elif clo_arg["closer"] == "log-normal":
                color="r"
            elif clo_arg["closer"] == "normal":
                color="m"
            else:
                raise Exception("unexpected closer: {0}".format(clo_arg["closer"]))
            lab = "closure: " + clo_arg["closer"]
            if not "multivariate" in clo_arg.keys():
                style = "-"
                lwd=2

            elif clo_arg["multivariate"]:
                lab = lab +"; multivariate"
                style = "--"
                lwd=2
            else:
                lab = lab +"; univariate"
                style = ":"
                lwd=2

            pl.plot(clo_arg["x_list"], clo_arg["y_list"],color=color, linewidth=lwd, linestyle=style, marker='o', label=lab)
            pl.legend()



        pl.savefig('figureB.pdf')
        pl.close()

        self.out_object = None


MyFigureA()
MyFigureB()
