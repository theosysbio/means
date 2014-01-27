#!/usr/bin/env python
import argparse
from datetime import datetime
import os
import difflib

ALLOWED_TESTS = ['mea', 'lna',
                 'simulation',
                 'inference']

MODELS = ['model_p53.txt', 'model_MM.txt', 'model_dimer.txt', 'model_Hes1.txt']

MEA_TEMPLATE = 'python runprogram.py --MEA --nMom={moments} --model={model_file} --ODEout=ODEout.tmp'
LNA_TEMPLATE = 'python runprogram.py --LNA --model={model_file} --ODEout=ODEout.tmp'
SIMULATION_TEMPLATE = 'python runprogram.py --MEA --nMom=3 --model={model_file} --compile {sundials_parameters} --timeparam={timeparam_file} --sim --simout={output_file} --ODEout=ODEout.tmp'
INFERENCE_TEMPLATE = 'python runprogram.py --MEA --model={model_file} --ODEout=ODEout.tmp --compile --library=library.tmp --timeparam={timeparam_file} --infer --data={dataset} --inferfile=inferout.tmp {sundials_parameters}'
SIMULATION_MODELS = ['MM', 'p53']
INFERENCE_MODELS = [('dimer', 'data_dimer_x40.txt')]


def create_options_parser():


    def _infer_sundials_parameters():
        # This is where sundials is stored on Mac OS X if homebrew was used to
        # install it
        if os.path.isfile('/usr/local/lib/libsundials_cvode.a'):
            return "--sd2=/usr/local/lib/ --sd1=/usr/local/include/"
        else:
            return None

    def _validate_tests(test):
        if not test in ALLOWED_TESTS:
            raise Exception('{0!r} not in the allowed test list: {1!r}'.format(test, ALLOWED_TESTS.keys()))
        return test

    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('Working directory', 'Location of working directories')
    group.add_argument('--inout-dir', help='Location of input/output directory',
                        default='../Inoutput')
    group.add_argument('--model-answers-dir', help="Location of model answers directory",
                        default='../Inoutput/model_answers')


    parser.add_argument('--build-reference-answers', action='store_true',
                        default='false', help='Generate reference results')

    parser.add_argument('-m', '--max-moment', type=int, help='Maximum moment to use',
                        default=2)

    parser.add_argument('tests', default=ALLOWED_TESTS, nargs='*',
                        help='Tests to run, must be one of {0}'.format(ALLOWED_TESTS),
                        type=_validate_tests)

    parser.add_argument('--sundials-parameters', help='Sundials parameters to use, '
                                                      'e.g. --sundials-paramteres="--sd1=/foo/bar --sd2=/bar/baz"',
                        default=_infer_sundials_parameters())

    return parser

class NoOutputGeneratedException(Exception):
    pass

class Test(object):

    command = None
    output_file = None
    expected_output_file = None
    comparison_function = None
    filter_function = None

    def __init__(self, command, output_file, expected_output_file,
                 comparison_function, filter_function=None):
        self.command = command
        self.output_file = output_file
        self.expected_output_file = expected_output_file
        self.comparison_function = comparison_function
        self.filter_function = filter_function

    def run_command(self):
        start_time = datetime.now()
        result = os.system(self.command)
        end_time = datetime.now()

        return result, end_time-start_time

    def _filter_output(self, output):
        if self.filter_function is None:
            return output
        else:
            return self.filter_function(output)

    def get_output(self):
        try:
            f = open(self.output_file, 'r')
        except IOError:
            raise NoOutputGeneratedException

        try:
            return self._filter_output(f.read())
        finally:
            f.close()

    def get_expected_output(self):
        try:
            f = open(self.expected_output_file, 'r')
        except IOError:
            raise Exception("Expected output file not found")

        try:
            return self._filter_output(f.read())
        finally:
            f.close()

    def compare_outputs(self):
        output, expected_output = self.get_output(), self.get_expected_output()
        return self.comparison_function(output, expected_output)

    def __str__(self):
        return '> {0}\n> Output: {1!r}, Expected Output: {2!r}'.format(self.command,
                                                                        self.output_file,
                                                                        self.expected_output_file)

def filter_time_taken(output):
    lines = output.splitlines()
    lines = filter(lambda x: not x.startswith('Time taken'), lines)
    return '\n'.join(lines)

def filter_input_file(output):
    lines = output.splitlines()
    lines = filter(lambda x: 'Input file:' not in x, lines)
    return '\n'.join(lines)

def diff_comparison(output, expected_output):
    if output == expected_output:
        return []
    else:
        differences = difflib.ndiff(output.splitlines(),
                                     expected_output.splitlines())
        return differences


def generate_tests_from_options(options):

    if 'mea' in options.tests:
        for model in MODELS:
            for moment in range(2, options.max_moment+1):
                yield Test(MEA_TEMPLATE.format(model_file=os.path.join(options.inout_dir, model),
                                               moments=moment),
                           os.path.join(options.inout_dir, 'ODEout.tmp'),
                           os.path.join(options.model_answers_dir, 'MEA{0}'.format(moment), model + '.out'),
                           diff_comparison,
                           filter_function=filter_time_taken)

    if 'lna' in options.tests:
        for model in MODELS:
            yield Test(LNA_TEMPLATE.format(model_file=os.path.join(options.inout_dir, model)),
                       os.path.join(options.inout_dir, 'ODEout.tmp'),
                       os.path.join(options.model_answers_dir, 'LNA', model + '.out'),
                       diff_comparison,
                       filter_function=filter_time_taken)

    if 'simulation' in options.tests:
        if options.sundials_parameters is None:
            raise Exception("Cannot run simulation tests as no sundials parameters specified")

        for model in SIMULATION_MODELS:
            output_file = 'simout_{0}.txt'.format(model)
            yield Test(SIMULATION_TEMPLATE.format(model_file=os.path.join(options.inout_dir, 'model_{0}.txt'.format(model)),
                                                  sundials_parameters=options.sundials_parameters,
                                                  timeparam_file=os.path.join(options.inout_dir, 'param_{0}.txt'.format(model)),
                                                  output_file=output_file),
                       os.path.join(options.inout_dir, output_file),
                       os.path.join(options.model_answers_dir, 'sim', output_file),
                       diff_comparison,
                       filter_function=filter_input_file)

    if 'inference' in options.tests:
        for model, dataset in INFERENCE_MODELS:
            yield Test(INFERENCE_TEMPLATE.format(model_file=os.path.join(options.inout_dir, 'model_{0}.txt'.format(model)),
                                                 sundials_parameters=options.sundials_parameters,
                                                 timeparam_file=os.path.join(options.inout_dir, 'param_{0}.txt'.format(model)),
                                                 dataset=dataset),
                       os.path.join(options.inout_dir, 'inferout.tmp'),
                       os.path.join(options.model_answers_dir, 'infer', 'infer_{0}.txt'.format(model)),
                       diff_comparison,
                       filter_function=filter_input_file)



def main():
    parser = create_options_parser()
    options = parser.parse_args()


    tests_to_run = generate_tests_from_options(options)

    for i, test in enumerate(tests_to_run):
        print '> Running test #{0}'.format(i+1)
        print test
        test.run_command()
        differences = test.compare_outputs()
        if not differences:
            print "> ALL OK"
            print
        else:
            print '> Test FAILED, here are the differences between files:'
            print '\n'.join(differences)
            break

if __name__ == '__main__':
    main()
