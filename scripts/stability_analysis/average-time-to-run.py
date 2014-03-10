import means
import means.examples
import numpy as np
from functools import wraps
import os
import cPickle as pickle
import timeit
import hashlib
import pandas as pd
from collections import defaultdict

DISK_CACHE_DIRECTORY = '.cache'

MODEL = means.examples.MODEL_P53
MAX_ORDERS = [1, 2, 3, 4, 5, 6]
CLOSURE_METHODS = ['normal', 'scalar', 'log-normal']

PARAMETERS = {'safe': [90, 0.002, 1.20, 1.1, 2.00, 0.96, 0.01],
              'unsafe': [90, 0.002, 2.10, 1.1, 0.80, 0.96, 0.01]}

INITIAL_CONDITIONS = [70, 30, 90]
TIMEPOINTS = np.arange(0, 40, 0.1)

SIMULATION_KWARGS_TO_TEST = [{'solver': solver} for solver in means.simulation.Simulation.supported_solvers()]
# I wonder if Newton iteration changes things
SIMULATION_KWARGS_TO_TEST.append({'solver': 'ode15s', 'iter': 'Newton'})
# Let' see how higher tolerance influences things as well
SIMULATION_KWARGS_TO_TEST.append({'solver': 'ode15s', 'rtol': 1e-6})
# Euler with small step size
SIMULATION_KWARGS_TO_TEST.append({'solver': 'euler', 'h': 0.001})

def disk_cached(function):

    if not os.path.exists(DISK_CACHE_DIRECTORY):
        os.mkdir(DISK_CACHE_DIRECTORY)

    def _filename(args, kwargs):
        key = pickle.dumps(list(args) + sorted(kwargs.items()), pickle.HIGHEST_PROTOCOL)
        key = hashlib.sha1(key).hexdigest()
        key = '.'.join([key, 'pkl'])
        return os.path.join(DISK_CACHE_DIRECTORY, key)

    def _store_answer_in_cache(filename, function, args, kwargs):
        answer = function(*args, **kwargs)
        with open(filename, 'w') as file_handle:
            pickle.dump(answer, file_handle)

        return answer

    @wraps(function)
    def f(*args, **kwargs):
        filename = _filename(args, kwargs)

        try:
            with open(filename, 'r') as file_handle:
                print 'Returning result for {0}(*{1!r}, **{2!r}) from cache'.format(function.__name__, args, kwargs)
                return pickle.load(file_handle)
        except IOError:
            return _store_answer_in_cache(filename, function, args, kwargs)

    return f

@disk_cached
def _get_problem(max_order, closure):
    return means.approximation.MomentExpansionApproximation(MODEL, max_order=max_order, closure=closure).run()

def _generate_problems_dict():
    import itertools

    print 'Generating problems'

    problems = {}
    for max_order, closure in itertools.product(MAX_ORDERS, CLOSURE_METHODS):
        if max_order < 2 and closure != 'zero':
            continue
        print 'max_order={0!r}, closure={1!r}'.format(max_order, closure)
        problem = _get_problem(max_order, closure)
        problems[max_order, closure] = problem

    return problems

def runtime_test_function(simulation, value, initial_conditions, timepoints):
    def f():
        try:
            simulation.simulate_system(value, initial_conditions, timepoints)
            return True
        except Exception:
            return False
    return f

@disk_cached
def _test_runtime_for(problem, number_of_runs=10, *args, **kwargs):
    RUNTIME_THRESHOLD = 3600 / 10.0

    simulation = means.simulation.Simulation(problem, *args, **kwargs)
    # Warm simulation instance up (cache the numerical evaluation routines)

    try:
        simulation.simulate_system(PARAMETERS['safe'], INITIAL_CONDITIONS, np.arange(0, 1, 0.5))
    except Exception:
        pass


    runtimes = {}
    for key, value in PARAMETERS.iteritems():
        print "Timing {0}".format(key)
        timer = timeit.Timer(runtime_test_function(simulation, value, INITIAL_CONDITIONS, TIMEPOINTS))
        runtime_one = timer.timeit(number=1)

        if runtime_one > RUNTIME_THRESHOLD:
            print "Runtime for one iteration was {0}, which is greater than {1}"  \
                  "not executing the remaining runs".format(runtime_one, RUNTIME_THRESHOLD)
            runtimes[key] = runtime_one
            continue

        runtime = timer.timeit(number=number_of_runs-1)
        runtime += runtime_one  # Add the first one again
        runtime /= float(number_of_runs)
        print "Timing {0} result: {1}s per iteration".format(key, runtime)
        runtimes[key] = runtime

    return runtimes

@disk_cached
def compute_runtimes():
    problems = _generate_problems_dict()

    runtimes = []

    for description, problem in problems.iteritems():
        max_order, closure = description

        for kwargs in SIMULATION_KWARGS_TO_TEST:
            kwargs_key = ', '.join(map(lambda x: '{0}={1!r}'.format(x[0], x[1]), sorted(kwargs.items())))
            print "Testing runtimes for {0}".format(kwargs_key)

            parameter_runtimes = _test_runtime_for(problem, **kwargs)
            for key, value in parameter_runtimes.iteritems():
                d = {'max_order': max_order,
                     'closure': closure}
                d.update(kwargs)
                d['parameter_set'] = key
                d['runtime'] = value

                runtimes.append(d)
    return pd.DataFrame(runtimes)

if __name__ == '__main__':
    compute_runtimes()



