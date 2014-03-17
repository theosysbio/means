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
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), required=True)

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
    yaml.dump(test_cases, args.output)
    args.output.close()

if __name__ == '__main__':
    main()



