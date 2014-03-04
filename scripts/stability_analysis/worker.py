import multiprocessing
import sys
import means
import means.examples
import numpy as np
import yaml

NUMBER_OF_PROCESSES = multiprocessing.cpu_count()
MODEL = means.examples.MODEL_P53

# An item that is added to the queue after all items processed. Allows the process to die.
NO_MORE_ITEMS_TO_PROCESS = 'no-more-items-to-process'
OUTPUT_DATA_DIR = '.data'

import cPickle as pickle
import os

TIMEPOINTS = np.arange(0, 40, 0.1)

def recursively_generate_parameters(parameters_to_simulate, parameter_index, current_parameter_vector):
    if parameter_index >= len(parameters_to_simulate):
        yield current_parameter_vector[:]
    else:

        for parameter_value in parameters_to_simulate[parameter_index]:
            # Change the ith value
            current_parameter_vector[parameter_index] = parameter_value
            # recurse
            for result in recursively_generate_parameters(parameters_to_simulate, parameter_index+1,
                                                          current_parameter_vector):
                yield result

def simulation_parameters(params_file):
    with open(params_file) as f:
        contents = f.read()

    parameters = yaml.load(contents)

    parameters_for_simulation = []
    for param in parameters['parameters_for_simulation']:
        try:
            param = [float(param)]
        except TypeError:
            if len(param) == 3:
                param = np.arange(*param)
            else:
                raise

        parameters_for_simulation.append(param)

    # parameters_for_simulation = [#np.arange(70, 90, 0.1),
    #                              [90],
    #                              [0.002],
    #                              #np.arange(1.2, 2.2, 0.01),
    #                              [1.7],
    #                              [1.1],
    #                              np.arange(0.8, 2, 0.05),
    #                              np.arange(0.8, 2, 0.05),
    #                              [0.01]]

    max_orders_for_simulation = map(int, parameters['max_orders'])

    current_parameters_list = [None] * len(parameters_for_simulation)

    for parameter_set in recursively_generate_parameters(parameters_for_simulation, 0, current_parameters_list):
        for max_order in max_orders_for_simulation:
            yield {'max_order': max_order,
                   'parameters': parameter_set,
                   'initial_conditions': [70, 30, 90]}

def _unique_filename(kwargs):
    import hashlib
    kwargs_as_sorted_list = sorted(kwargs.items())
    payload = pickle.dumps(kwargs_as_sorted_list, pickle.HIGHEST_PROTOCOL)

    sha_hash = hashlib.sha1(payload).hexdigest()
    filename = '.'.join([sha_hash, 'pickle'])

    return os.path.join(OUTPUT_DATA_DIR, filename)

def process_f(queue):
    problems = {}

    while True:
        original_kwargs = queue.get(block=True)

        if original_kwargs == NO_MORE_ITEMS_TO_PROCESS:
            print 'Got NO_MORE_ITEMS_TO_PROCESS. Dying'
            break

        filename = _unique_filename(original_kwargs)
        if os.path.exists(filename):
            continue

        kwargs = original_kwargs.copy()
        max_order = kwargs.pop('max_order')
        try:
            problem = problems[max_order]
        except KeyError:
            print 'Problem for max_order={0!r} not cached yet, computing it'.format(max_order)
            problem = means.approximation.MomentExpansionApproximation(MODEL, max_order=max_order).run()
            problems[max_order] = problem

        #print 'Simulating for {0!r}'.format(kwargs)
        simulation = means.simulation.Simulation(problem=problem)
        exception_caught = None
        try:
            trajectories = simulation.simulate_system(timepoints=TIMEPOINTS, **kwargs)
        except Exception as e:
            exception_caught = e
            trajectories = None

        with open(filename, 'w') as f:

            pickle.dump({'kwargs': original_kwargs,
                         'trajectories': trajectories,
                         'exception': exception_caught}, f)


def main():
    if not os.path.exists(OUTPUT_DATA_DIR):
        os.mkdir(OUTPUT_DATA_DIR)

    param_file = sys.argv[1]

    try:
        mode = sys.argv[2]
    except IndexError:
        mode = 'work'

    if mode == 'count':
        print 'Count only mode:'
        print len(list(simulation_parameters(param_file)))
        return
    elif mode != 'work':
        raise Exception("Unknown mode {0!r} provided".format(mode))

    queue = multiprocessing.Queue()

    processes = []
    for __ in xrange(NUMBER_OF_PROCESSES):
        process = multiprocessing.Process(target=process_f, args=(queue,))
        processes.append(process)

    # Start processes
    for process in processes:
        process.start()

    count = 0
    for parameter_set in simulation_parameters(param_file):
        count += 1
        queue.put(parameter_set)

    # Add the special markers saying no more items to process for each of the processes
    for __ in processes:
        queue.put(NO_MORE_ITEMS_TO_PROCESS)

    # This says wait for all processes to finish before going to next line
    for process in processes:
        process.join()

    print 'Successfully processed {0} parameters'.format(count)


if __name__ == '__main__':
    main()