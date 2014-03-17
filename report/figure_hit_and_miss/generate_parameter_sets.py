import argparse
import numpy as np
import cPickle as pickle
import hashlib
import yaml

def hash(parameters, initial_conditions, timepoints, kwargs):
    payload = (parameters, initial_conditions, timepoints, sorted(kwargs.items()))
    dump = pickle.dumps(payload)
    hash_ = hashlib.md5(dump).hexdigest()
    return hash_

def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--per-file', type=int, default=0, help='If set, would create multiple files,'
                                                                ' each containing specified number of items')

    return parser

def generate_test_cases():

    test_cases = []

    for c_2 in np.arange(0.7, 2.7, 0.1):
        c_2 = round(c_2, 4)
        for c_4 in np.arange(0.7, 2.7, 0.1):
            c_4 = round(c_4, 4)
            parameters = [90, 0.002, c_2, 1.1, c_4, 0.96, 0.01]
            initial_conditions = [70, 30, 90]
            kwargs = {'solver': 'ode15s'}
            timepoints = [0, 40, 0.1]

            filename = hash(parameters, initial_conditions, timepoints, kwargs)
            output = 'p53-{0}.yml'.format(filename)

            data = {'parameters': parameters,
                    'initial_conditions': initial_conditions,
                    'simulation_kwargs': kwargs,
                    'timepoints': timepoints,
                    'output': output}
            test_cases.append(data)
    return test_cases

def main():
    args = _parser().parse_args()
    test_cases = generate_test_cases()

    if args.per_file <= 0:
        splits = [test_cases]
    else:
        splits = []
        while len(test_cases) > args.per_file:
            splits.append(test_cases[:args.per_file])
            test_cases = test_cases[args.per_file:]

    for i, split in enumerate(splits):
        filename = args.output
        if i > 0:
            filename += '.{0}'.format(i)
        with open(filename, 'w') as f:
            yaml.dump(split, f)

if __name__ == '__main__':
    main()



