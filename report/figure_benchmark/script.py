import sys
sys.path.append("../utils")
from report_unit import ReportUnit
import pylab as pl
import pickle
import math

class MyFigure(ReportUnit):
    def __init__(self):
        super(MyFigure, self).__init__()

    def run(self):
        with open("../data_benchmark.pickle") as f:
            dic = pickle.load(f)
        pl.figure(figsize=(16.0, 9.0))
        pl.ylabel('log_10(dt) (s)')
        pl.xlabel('number of ODEs')
        for d in dic:
            pl.plot(d["n_eq"], d["dt"], linewidth=2.5, linestyle='--', marker='o', label=d["legend"])
        pl.legend(loc='upper right')
        #pl.show()
        pl.savefig('figure.pdf')

        self.out_object = None



MyFigure()