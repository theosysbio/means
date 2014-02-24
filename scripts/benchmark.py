import subprocess
import time
import sys

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


to_benchmark = [
    #{"git_tag":"means_no_optims", "legend":"my legend", "function":benchmark_means, "test_from": 2, "test_up_to": 4},
    {"git_tag":"no_simplify_and_cache_diff", "legend":"my legend", "function":benchmark_means, "test_from": 2, "test_up_to": 5},
    {"git_tag":"use_xreplace", "legend":"my legend", "function":benchmark_means, "test_from": 2, "test_up_to": 5},
    {"git_tag":"only_necessary_moms", "legend":"my legend", "function":benchmark_means, "test_from": 1, "test_up_to": 5},
    {"git_tag":"use_quick_solve", "legend":"my legend", "function":benchmark_means, "test_from": 1, "test_up_to": 6}
    #{"git_tag":"custom_diff", "legend":"my legend", "function":benchmark_means, "test_from": 1, "test_up_to": 6},
]
for tb in to_benchmark:
    tb["dt"] = []
    tb["n_eq"] = []

#means_no_optims no_simplify_and_cache_diff use_xreplace only_necessary_moms use_quick_solve custom_diff
highest_max_order = max([tb["test_up_to"] for tb in to_benchmark])

try:
    for max_order in range(1,highest_max_order + 1):
        print "# -------------------------"
        print "# Testing for max_order = {0}".format(max_order)
        for tb in to_benchmark:

            if tb["test_from"] <= max_order <= tb["test_up_to"]:
                process = subprocess.Popen(['git', 'checkout', tb["git_tag"]], stderr=subprocess.PIPE,  )
                out, err = process.communicate()
                # if err:
                #     print "??"
                #     sys.stderr("Failed to switch branch, the error was:")
                #     raise Exception

                time.sleep(3)
                n_eq, dt = tb["function"](max_order)
                print tb["git_tag"],dt,n_eq
                tb["dt"].append(dt)
                tb["n_eq"].append(dt)

finally:
    # We ensure we switch back to our branch
    subprocess.Popen(['git', 'checkout', GIT_HEAD]).communicate()
    exit(0)

