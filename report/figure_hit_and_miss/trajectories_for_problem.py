import os
import traceback
import yaml
import means
import means.examples
import argparse
import numpy as np
from datetime import datetime
import multiprocessing

def _deserialisable_problem(filename):
    return means.ODEProblem.from_file(filename)

def _existing_dir(directory):
    if os.path.exists(directory):
        if not os.path.isdir(directory):
            raise ValueError('Directory provided {0!r} is not a directory'.format(directory))
    else:
        os.mkdir(directory)

    return directory

def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', help='Problem to simulate', type=_deserialisable_problem)
    parser.add_argument('input', help='YAML file containing the inputs for generation,'
                                      'as well as their output files and solver options',
                        type=argparse.FileType('r'))
    parser.add_argument('-d', '--directory', help='Directory to put output files', default='.', type=_existing_dir)

    parser.add_argument('--processes', help="Number of processes ot use", type=int, default=1)

    return parser

def _pool_initialiser(problem_local, directory_local):
    global problem, directory
    problem = problem_local
    # Warm-up the problem:
    __ = problem.right_hand_side_as_function

    directory = directory_local

def _process_task(task):
    global problem, directory

    parameters = task['parameters']
    initial_conditions = task['initial_conditions']
    timepoints = task['timepoints']
    if len(timepoints) == 3:
        timepoints = np.arange(*timepoints)

    output = os.path.join(directory, task['output'])
    if os.path.exists(output):
        print 'Skipping generation of {0} as it already exists'.format(output)
        return

    simulation_kwargs = task['simulation_kwargs']

    simulation = means.Simulation(problem, **simulation_kwargs)

    exception = None
    start = datetime.now()
    try:
        trajectories = simulation.simulate_system(parameters=parameters,
                                                  initial_conditions=initial_conditions,
                                                  timepoints=timepoints)
    except means.SolverException as e:
        exception = e
        trajectories = None

    end = datetime.now()
    time_taken = (end-start).total_seconds()

    # Leave only first order trajectories
    if trajectories:
        means_ = filter(lambda x: isinstance(x.description, means.Moment) and x.description.order == 1,
                        trajectories)
    else:
        means_ = None
    means.io.to_file({'trajectories': means_,
                      'exception': exception,
                      'time_taken': time_taken,
                      'task': task}, output)


def _wrapped_process_task(task):
    try:
        _process_task(task)
    except:
        traceback.print_exc()
        raise

def main():

    parser = _parser()
    options = parser.parse_args()
    directory = options.directory

    tasks = yaml.load(options.input)

    problem = options.problem

    if options.processes > 1:
        pool = multiprocessing.Pool(processes=options.processes,
                                    initializer=_pool_initialiser,
                                    initargs=(problem, directory))
        __ = pool.map(_wrapped_process_task, tasks)
        pool.close()
    else:
        _pool_initialiser(problem, directory)
        __ = map(_wrapped_process_task, tasks)
if __name__ == '__main__':
    main()