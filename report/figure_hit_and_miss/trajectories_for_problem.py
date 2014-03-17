import means
import means.examples
import argparse

def _deserialisable_problem(filename):
    return means.ODEProblem.from_file(filename)

def _parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--problem', help='Problem to simulate', type=_deserialisable_problem)
    parser.add_argument('--parameters', help='Parameters for the generation', type=float, nargs='+')
    parser.add_argument('--initial-conditions', help='Initial conditions for generation')


    return parser

def main():
    parser = _parser()
    options = parser.parse_args()

    print options.problem
    print options.parameters
    print options.initial-conditions



if __name__ == '__main__':
    main()