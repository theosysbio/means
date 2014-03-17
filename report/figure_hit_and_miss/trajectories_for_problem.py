import os
import yaml
import means
import means.examples
import argparse
import numpy as np
from datetime import datetime

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

    return parser

def main():

    parser = _parser()
    options = parser.parse_args()
    directory = options.directory

    tasks = yaml.load(options.input)

    problem = options.problem

    # Warm-up the problem:
    problem.right_hand_side_as_function

    for task in tasks:
        parameters = task['parameters']
        initial_conditions = task['initial_conditions']
        timepoints = task['timepoints']
        if len(timepoints) == 3:
            timepoints = np.arange(*timepoints)

        output = os.path.join(directory, task['output'])

        if os.path.exists(output):
            print 'Skipping generation of {0} as it already exists'.format(output)
            continue

        simulation_kwargs = task['simulation_kwargs']

        simulation = means.Simulation(options.problem, **simulation_kwargs)

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
        means_ = filter(lambda x: isinstance(x.description, means.Moment) and x.description.order == 1,
                        trajectories)
        means.io.to_file({'trajectories': means_,
                          'exception': exception,
                          'time_taken': time_taken}, output)

if __name__ == '__main__':
    main()