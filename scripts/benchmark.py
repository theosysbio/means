import subprocess
import time
import pylab as pl
# from ggplot import ggplot, aes
# from ggplot.geoms import *
# import pandas as pd

GIT_HEAD = "mea_performance"

def benchmark_means(max_order):
    str="\n".join([
        "import time",
        "from means.approximation.mea import MomentExpansionApproximation",
        "from means.examples.sample_models import MODEL_P53",
        "t0 = time.time()",
        "pb = MomentExpansionApproximation(MODEL_P53, %i, 'log-normal').run()",
        "print '{0}, {1}'.format(pb.number_of_equations, time.time() - t0)"
    ])

    script = str % max_order

    process = subprocess.Popen(['python', '-c', script], stdout=subprocess.PIPE)
    out, err = process.communicate()
    n_eqs, t = out.rstrip().split(",")
    return int(n_eqs), round(float(t),3)

# def plot_all_ggplot(df):
#     print ggplot(df, aes(x="n_eq", y="time", group="tag", colour="tag")) +\
#     geom_line() + geom_point()


def plot_all(dic):
    pl.figure()
    pl.ylabel('time(s)')
    pl.xlabel('number of ODEs')
    for d in dic:
        pl.plot(d["n_eq"], d["dt"], linewidth=2.5, linestyle='--', marker='o',label=d["legend"])
        print zip(d["n_eq"], d["dt"])
    pl.legend(loc='upper left')
    pl.show()




to_benchmark = [
    #{"git_tag":"means_no_optims", "legend":"my legend", "function":benchmark_means, "test_from": 2, "test_up_to": 4, "dt":[], "n_eq":[]},
    # {"git_tag":"no_simplify_and_cache_diff", "legend":"`simplify()` has been removed.", "function":benchmark_means, "test_from": 2, "test_up_to": 5, "dt":[], "n_eq":[]},
    # {"git_tag":"use_xreplace", "legend":"`xreplace()` is being used instead of `substitute()`", "function":benchmark_means, "test_from": 2, "test_up_to": 5, "dt":[], "n_eq":[]},
    {"git_tag":"only_necessary_moms", "legend":"we do not remove highest order moments", "function":benchmark_means, "test_from": 1, "test_up_to": 5, "dt":[], "n_eq":[]},
    {"git_tag":"use_quick_solve", "legend":"use custom function instead of `solve`", "function":benchmark_means, "test_from": 1, "test_up_to": 6, "dt":[], "n_eq":[]}
    #{"git_tag":"custom_diff", "legend":"my legend", "function":benchmark_means, "test_from": 1, "test_up_to": 6, "dt":[], "n_eq":[]},
]


#means_no_optims no_simplify_and_cache_diff use_xreplace only_necessary_moms use_quick_solve custom_diff
highest_max_order = max([tb["test_up_to"] for tb in to_benchmark])

df = pd.DataFrame(columns=("tag", "n_eq", "time"))
try:
    for max_order in range(1,highest_max_order + 1):
        print "# -------------------------"
        print "# Testing for max_order = {0}".format(max_order)
        for tb in to_benchmark:

            if tb["test_from"] <= max_order <= tb["test_up_to"]:
                process = subprocess.Popen(['git', 'checkout', tb["git_tag"]], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                out, err = process.communicate()
                if process.returncode != 0 :
                    print err
                    exit(1)
                time.sleep(1)
                n_eq, dt = tb["function"](max_order)
                tb["dt"].append(dt)
                tb["n_eq"].append(n_eq)

                # row = pd.DataFrame([dict(tag=tb["git_tag"], n_eq=n_eq, time=dt), ])
                # df = df.append(row, ignore_index=True)

except KeyboardInterrupt:
    pass
finally:
    subprocess.Popen(['git', 'checkout', GIT_HEAD]).communicate()
    time.sleep(1)
    
# We ensure we switch back to our branch
plot_all(to_benchmark)
exit(0)


