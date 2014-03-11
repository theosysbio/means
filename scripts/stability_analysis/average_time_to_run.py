import means
import means.examples
import numpy as np
from functools import wraps
import os
import cPickle as pickle
import timeit
import hashlib
import pandas as pd
import multiprocessing

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

# Let's make it run in parallel, this is nonsense otherwise
N_PROCESSES = max(multiprocessing.cpu_count(), 9)  # 9, not 8 as we have 18 problems

def _try_hipchat_notify(message, color='gray', *args, **kwargs):
    print message

    try:
        import hipchat
    except ImportError:
        print 'Cannot notify to hipchat, do `pip install python-simple-hipchat`'
        return

    try:
        hipchat_config = means.io.from_file('hipchat_config.yml')
        room = hipchat_config['room']
        token = hipchat_config['token']

    except (IOError, KeyError):
        print 'No hipchat config provided, put `token` and `room_name` into hipchat_config.yml'
        return

    try:
        hipster = hipchat.HipChat(token=token)
        hipster.message_room(room, os.path.basename(__file__), message, color=color, *args, **kwargs)
    except Exception as e:
        print 'Hipchat notification failed: {0!r}'.format(e)
        return

    return

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
        if max_order < 2 and closure != 'scalar':
            continue
        print 'max_order={0!r}, closure={1!r}'.format(max_order, closure)
        problem = _get_problem(max_order, closure)
        problems[max_order, closure] = problem

    _try_hipchat_notify('Finished generating problems. Let the games begin')

    return problems

def runtime_test_function(simulation, value, initial_conditions, timepoints):
    def f():
        try:
            simulation.simulate_system(value, initial_conditions, timepoints)
            return True
        except means.SolverException:
            return False
        except Exception as e:
            print 'Got exception {0!r} when timing'.format(e)
            return False
    return f

def _dict_to_str(dict_):
    return '/'.join(['{0}={1!r}'.format(x, y) for x, y in sorted(dict_.items())])

@disk_cached
def _test_runtime_for(problem, number_of_runs=10, **kwargs):
    RUNTIME_THRESHOLD = 3600 / 10.0

    simulation = means.simulation.Simulation(problem, **kwargs)
    # Warm simulation instance up (cache the numerical evaluation routines)

    rhs_as_function = problem.right_hand_side_as_function

    runtimes = {}
    for key, value in PARAMETERS.iteritems():
        print "{1}\t{0}: pending".format(key, _dict_to_str(kwargs))
        timer = timeit.Timer(runtime_test_function(simulation, value, INITIAL_CONDITIONS, TIMEPOINTS))
        runtime_one = timer.timeit(number=1)
        print "{2}\t{0}: {1}s first iteration".format(key, runtime_one, _dict_to_str(kwargs))
        if runtime_one > RUNTIME_THRESHOLD:
            print "Runtime for one iteration was {0}, which is greater than {1}"  \
                  "not executing the remaining runs".format(runtime_one, RUNTIME_THRESHOLD)
            runtimes[key] = runtime_one
            continue

        curr_runtimes = timer.repeat(repeat=number_of_runs-1, number=1)
        curr_runtimes.append(runtime_one)

        print "{2}\t{0}: avg: {1}s per iteration".format(key, np.mean(curr_runtimes), _dict_to_str(kwargs))
        runtimes[key] = curr_runtimes

    return runtimes

def compute_runtimes_for_problem(argument):

    # Map joins the arguments into a tuple
    description, problem = argument

    max_order, closure = description
    problem_runtimes = []
    message_runtimes = []

    for j, kwargs in enumerate(SIMULATION_KWARGS_TO_TEST):

        parameter_runtimes = _test_runtime_for(problem, **kwargs)
        for key, value in parameter_runtimes.iteritems():
            for runtime in value:
                d = {'max_order': max_order,
                     'closure': closure}
                d.update(kwargs)
                d['parameter_set'] = key
                d['runtime'] = runtime

                problem_runtimes.append(d)
            message_runtimes.append(", ".join(["{0}={1}".format(x, y) for x, y in sorted(d.items())
                                               if x not in ['closure', 'max_order']]))

    message = "Simulations for {0!r} ave now finished. " \
              "Results: \n{1}".format(description, '\n'.join(message_runtimes))
    _try_hipchat_notify(message)

    return problem_runtimes


@disk_cached
def compute_runtimes():
    problems = _generate_problems_dict()

    runtimes = []

    pool = multiprocessing.Pool(N_PROCESSES)
    lists_of_runtimes = pool.map(compute_runtimes_for_problem, sorted(problems.items(), reverse=True))

    pool.close()

    for runtime in lists_of_runtimes:
        runtimes.extend(runtime)

    return pd.DataFrame(runtimes)

if __name__ == '__main__':
    compute_runtimes()



