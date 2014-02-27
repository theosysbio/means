import sys
sys.path.append("../utils")
from report_unit import ReportUnit
import subprocess
import time
import math

MATLAB_PKG_DIR="/home/quentin/matlab/momentexpansion_matlab/equations"
GIT_HEAD = "mea_performance"



class MyFigure(ReportUnit):
    def __init__(self):
        super(MyFigure, self).__init__()

    def run(self):
        to_benchmark = [
            #{"git_tag": None, "legend":"matlab package", "function":self.benchmark_matlab, "test_from": 1, "test_up_to": 3, "dt":[], "n_eq":[]},
            {"git_tag":"means_no_optims", "legend":"means, no optimisation", "function":self.benchmark_means, "test_from": 2, "test_up_to": 4, "dt":[], "n_eq":[]},
            {"git_tag":"no_simplify_and_cache_diff", "legend":"`simplify()` has been removed.", "function":self.benchmark_means, "test_from": 2, "test_up_to": 5, "dt":[], "n_eq":[]},
            {"git_tag":"use_xreplace", "legend":"`xreplace()` is being used instead of `substitute()`", "function":self.benchmark_means, "test_from": 2, "test_up_to": 5, "dt":[], "n_eq":[]},
            {"git_tag": "only_necessary_moms", "legend":"we do not remove highest order moments", "function":self.benchmark_means, "test_from": 1, "test_up_to": 5, "dt":[], "n_eq":[]},
            {"git_tag":"use_quick_solve", "legend":"use custom function instead of `solve`", "function":self.benchmark_means, "test_from": 1, "test_up_to": 6, "dt":[], "n_eq":[]},
            {"git_tag":"custom_diff", "legend":"use custom differentiation", "function":self.benchmark_means, "test_from": 1, "test_up_to": 6, "dt":[], "n_eq":[]}
        ]

        #means_no_optims no_simplify_and_cache_diff use_xreplace only_necessary_moms use_quick_solve custom_diff
        highest_max_order = max([tb["test_up_to"] for tb in to_benchmark])
        #df = pd.DataFrame(columns=("tag", "n_eq", "time"))
        try:
            for max_order in range(1, highest_max_order + 1):
                print "# -------------------------"
                print "# Testing for max_order = {0}".format(max_order)
                for tb in to_benchmark:

                    if tb["test_from"] <= max_order <= tb["test_up_to"]:
                        self.git_swing(tb["git_tag"])
                        t0 = time.time()
                        n_eq = tb["function"](max_order)
                        logdt = round(math.log10(time.time() - t0),4)
                        tb["dt"].append(logdt)
                        tb["n_eq"].append(n_eq)
                        print tb["git_tag"], n_eq, logdt


        finally:
            print "wth"
            subprocess.Popen(['git', 'checkout', GIT_HEAD]).communicate()
            time.sleep(1)
            print "wth"
            self.out_object = to_benchmark

    def git_swing(self, branch=GIT_HEAD):
        if branch:
            process = subprocess.Popen(['git', 'checkout', branch], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            out, err = process.communicate()
            time.sleep(1)
            if process.returncode != 0:
                print "GIT ERROR:"
                print err
                exit(1)

    def benchmark_means(self, max_order):
        str="\n".join([
            "from means.approximation.mea import MomentExpansionApproximation",
            "from means.examples.sample_models import MODEL_P53",
            "pb = MomentExpansionApproximation(MODEL_P53, %i, 'log-normal').run()",
            "print '{0}'.format(pb.number_of_equations)"
        ])

        script = str % max_order

        process = subprocess.Popen(['python', '-c', script], stdout=subprocess.PIPE)
        out, err = process.communicate()
        return int(out.rstrip())


    def benchmark_matlab(self, max_order):


        str=";".join([
            "cd('{0}')".format(MATLAB_PKG_DIR),
            "[MFK,M,CentralMoments] = MFK_create_symbolic_automatic_lognormal(%i, 1)",
            "n_eq = length(MFK)",
            "disp(n_eq);"
            "exit();"
        ])

        script = (str % max_order)
        process = subprocess.Popen(['matlab', '-nodesktop', '-nodisplay', '-r', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()

        out_str = process.stdout.read()

        for os in out_str.split("\n"):
            s = os.strip().rstrip()
            try:
                res = int(s)
            except:
                pass
        # return the last int
        return res

MyFigure()