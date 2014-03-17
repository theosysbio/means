import means
import means.examples
import argparse
import numpy as np
from datetime import datetime

def _deserialisable_problem(filename):
    return means.ODEProblem.from_file(filename)

def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help="Output file to write trajectories to", required=True,
                        type=argparse.FileType('w'))
    parser.add_argument('problem', help='Problem to simulate', type=_deserialisable_problem)
    parser.add_argument('--parameters', help='Parameters for the generation',
                        type=float, nargs='+', required=True)
    parser.add_argument('--initial-conditions', help='Initial conditions for generation',
                        type=float, nargs='+', required=True)
    parser.add_argument('--last-timepoint', help="Last timepoint to simulate to, "
                                                 "the simulation would run from 0 to last_timepoint with step size 0.1",
                        type=float, required=True)

    solver_options_group = parser.add_argument_group('Solver Options')
    solver_options_group.add_argument('-s', '--solver', help="Solver to use", default='ode15s')

    return parser

def main():
    parser = _parser()
    options = parser.parse_args()

    simulation = means.Simulation(options.problem, solver=options.solver)
    timepoints = np.arange(0, options.last_timepoint, 0.1)
    exception = None
    start = datetime.now()
    try:
        trajectories = simulation.simulate_system(parameters=options.parameters,
                                                  initial_conditions=options.initial_conditions,
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
                      'time_taken': time_taken}, options.output)

    options.output.close()


if __name__ == '__main__':
    main()