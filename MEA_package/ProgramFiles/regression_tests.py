#!/usr/bin/env python
import argparse
from datetime import datetime
import os
import difflib
import subprocess
import traceback
import numpy as np
import scipy.spatial.distance
import ode_problem
import sympyhelpers


ALLOWED_TESTS = ['mea', 'lna',
                 'simulation',
                 'inference',
                 'inference-with-restarts',
                 'inference-with-distributions']

MODELS = ['model_p53.txt', 'model_MM.txt', 'model_dimer.txt', 'model_Hes1.txt']

MEA_TEMPLATE = 'python runprogram.py --MEA --nMom={moments} --model={model_file} --ODEout=ODEout.tmp'
LNA_TEMPLATE = 'python runprogram.py --LNA --model={model_file} --ODEout=ODEout.tmp'
SIMULATION_TEMPLATE = 'python runprogram.py --{method} --nMom=3 --model={model_file} --compile {sundials_parameters} --timeparam={timeparam_file} --sim --simout={output_file} --ODEout=ODEout.tmp'
INFERENCE_TEMPLATE = 'python runprogram.py --MEA --model={model_file} --ODEout=ODEout.tmp --compile --library=library.tmp --timeparam={timeparam_file} --infer --data={dataset} --inferfile=inferout.tmp {sundials_parameters}'
INFERENCE_WITH_RESTARTS_TEMPLATE = 'python runprogram.py --MEA --model={model_file} --ODEout=ODEout.tmp --compile --library=library.tmp --timeparam={timeparam_file} --infer --data={dataset} --inferfile=inferout.restarts.tmp --restart --nRestart=10 {sundials_parameters}'
INFERENCE_WITH_DISTRIBUTIONS_TEMPLATE = 'python runprogram.py --MEA --model={model_file} --ODEout=ODEout.tmp --compile --library=library.tmp --timeparam={timeparam_file} --infer --data={dataset} --inferfile=inferout.tmp --limit --pdf={distribution} {restart_params} {sundials_parameters}'
SIMULATION_MODELS = ['MM', 'p53']
INFERENCE_MODELS = [('dimer', 'data_dimer_x40.txt', 'infer_dimer_x40.txt'),
                    ('dimer', 'data_dimer_x40_mean.txt', 'infer_dimer_x40_mean.txt'),
                    ('Hes1', 'data_Hes1.txt', 'infer_Hes1.txt')]
INFERENCE_WITH_RESTARTS_MODELS = [('dimer', 'data_dimer_x40.txt', 'infer_dimer_x40.txt', 0.0015, 1e-6),
                                  # This one is a bit bad at param inference, so lots of slack distance constraints
                                  ('dimer', 'data_dimer_x40_mean.txt', 'infer_dimer_x40_mean.txt', 0.1, 1e-6),
                                  ('Hes1', 'data_Hes1.txt', 'infer_Hes1.txt', 2, 0.6)] # Let's give lots of slack for this one

INFERENCE_DISTRIBUTIONS = ['gamma', 'normal', 'lognormal']
INFERENCE_WITH_DISTRIBUTIONS_MODELS = [('dimer', 'data_dimer_x40_mean.txt', 'infer_dimer_x40_mean_{0}.txt', 1e-6, 1e-6),
                                       ('Hes1', 'data_Hes1.txt', 'infer_Hes1_{0}.txt', 1e-6, 1e-6)]

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
            raise Exception('{0!r} not in the allowed test list: {1!r}'.format(test, ALLOWED_TESTS))
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

    parser.add_argument('--xunit', help='Return output in xunit format (parseable by Jenkins)',
                        default=False, action='store_true')

    return parser

class NoOutputGeneratedException(Exception):
    pass

class Test(object):

    name = None
    command = None
    output_file = None
    expected_output_file = None
    comparison_function = None
    filter_function = None

    def __init__(self, name, command, output_file, expected_output_file,
                 comparison_function, filter_function=None):
        self.name = name
        self.command = command
        self.output_file = output_file
        self.expected_output_file = expected_output_file
        self.comparison_function = comparison_function
        self.filter_function = filter_function

    def cleanup(self):
        # Cleanup
        try:
            os.remove(self.output_file)
        except OSError:
            pass

    def run_command(self):
        start_time = datetime.now()
        proc = subprocess.Popen([self.command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        end_time = datetime.now()

        return out, err, end_time-start_time

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

def compare_ode_problems(output, expected_output):


    expected_problem = ode_problem.parse_model(expected_output, from_string=True)
    result_problem = ode_problem.parse_model(output, from_string=True)

    expected_mom_dic = expected_problem.moment_dic
    result_mom_dic = result_problem.moment_dic

    expected_mom_keys = set(expected_mom_dic .keys())
    result_mom_keys = set(result_mom_dic.keys())


    # ensure we have the same keys
    if len(result_mom_keys - expected_mom_keys) != 0:
        return "Difference in the moments: \nexpected=\n%s\nresult=\n%s" % (str(expected_mom_keys),
                                                                            str(result_mom_keys))

    expected_rhs = expected_problem.right_hand_side
    result_rhs = result_problem.right_hand_side

    for e,r in zip(expected_rhs, result_rhs):
        if not sympyhelpers.deep_compare_expressions(e,r):
            return "different rhs equations!! \nexpected=\n%s\nresult=\n%s" %(str(e),str(r))

    if expected_problem.left_hand_side != result_problem.left_hand_side:
        return "different lhs equations!! \nexpected=\n%s\nresult=\n%s" % (str(expected_problem.left_hand_side),
                                                                           str(result_problem.left_hand_side))

def compare_tsv_with_float_epsilon(output, expected_output, epsilon=1e-7):
    def generate_dictionary_of_header_columns(lines):
        d = {}
        for line in lines:
            columns = line.split('\t')
            d[columns[0]] = columns[1:]
        return d
    # Do nothing if things equal
    if output == expected_output:
        return []

    differences = []

    output_lines = output.splitlines()
    expected_output_lines = expected_output.splitlines()

    left_columns_dict = generate_dictionary_of_header_columns(output_lines)
    right_columns_dict = generate_dictionary_of_header_columns(expected_output_lines)

    for right_header, right_columns in right_columns_dict.iteritems():

        try:
            left_columns = left_columns_dict[right_header]
        except KeyError:
            differences.append('Column with header {0} does not exist in output file'.format(right_header))
            continue

        if len(left_columns) != len(right_columns):
            differences.append('Number of columns does not match')
            equal = False
        else:
            equal = True
            for output_column, expected_output_column in zip(left_columns, right_columns):
                # Check for strict equality first
                if output_column == expected_output_column:
                    continue

                # Convert to floating point
                try:
                    float_o_c, float_e_o_c = float(output_column), float(expected_output_column)
                except ValueError:
                    # If conversion failed, and we already know that the lines aren't equal,
                    # conclude that the lines aren't equal
                    differences.append('DIFFERENCE: {0!r} not equal to {1!r} and aren\'t floats'.format(output_column,
                                                                                                    expected_output_column))
                    equal = False
                    break

                # Check if floats differ within epsilon
                if abs(float_o_c - float_e_o_c) > epsilon:
                    differences.append('FLOAT DIFFERENCE: {0} different from {1} '
                                       'by more than {2}'.format(float_o_c, float_e_o_c, epsilon))
                    equal = False
                    break

        if not equal:
            differences.append('Lines: ')
            differences.append('\t'.join([right_header] + left_columns))
            differences.append('\t'.join([right_header] + right_columns))

    return differences

def parameter_and_distance_comparisons(allowed_difference_between_top_distances=1e-6,
                                       allowed_difference_between_parameters=1e-14):

    def f(output, expected_output):

        def parse_distance(line):
            """
            Parses the floating point distance from a string like
            "Distance at minimum: 0.00257923702193"
            :param line:
            :return:
            """
            # More hacky, but quicker to write than regexp:
            return float(line.split(':')[1])

        def parse_parameters(line):
            """
            Parses the parameters from a string like:
            Optimised parameters:	[0.00012706267823174886, 0.089218889489651565, 301.05734997831519]
            Note that this uses `eval` so very unsafe, but hey...
            :param line:
            :return:
            """
            return eval(line.split(':')[1].strip())


        output_distance_lines = filter(lambda x: 'Distance at minimum' in x, output.splitlines())
        expected_output_distance_lines = filter(lambda x: 'Distance at minimum' in x, expected_output.splitlines())

        output_parameters_lines = filter(lambda x: 'Optimised parameters' in x, output.splitlines())
        expected_output_parameters_lines = filter(lambda x: 'Optimised parameters' in x, expected_output.splitlines())

        differences = []

        if len(output_distance_lines) != len(expected_output_distance_lines):
            differences.append('Number "distance at minimum" lines do not match in the outputs. '
                               'Got {0}, expected: {1}'.format(len(output_distance_lines),
                                                               len(expected_output_distance_lines)))
        else:
            min_distance_output = parse_distance(output_distance_lines[0])
            min_distance_expected_output = parse_distance(expected_output_distance_lines[0])

            if abs(min_distance_output - min_distance_expected_output) > allowed_difference_between_top_distances:
                differences.append("The minimum distances between the expected output and the actual one "
                                   "differ by more than {0}".format(allowed_difference_between_top_distances))
                differences.append("Got: {0}".format(min_distance_output))
                differences.append("Expected: {0}".format(min_distance_expected_output))

            best_parameters_o = np.array(parse_parameters(output_parameters_lines[0]))
            best_parameters_e = np.array(parse_parameters(expected_output_parameters_lines[0]))

            distance = scipy.spatial.distance.cosine(best_parameters_o, best_parameters_e)
            # TODO: this should be some sort of significance test
            if distance > allowed_difference_between_parameters:
                differences.append("Minimum distances between the expected parameters and actual ones "
                                   "differ by more than {0}".format(allowed_difference_between_parameters))

                differences.append("Got: {0}".format(best_parameters_o))
                differences.append("Expected: {0}".format(best_parameters_e))
                differences.append("Distance: {0}".format(distance))

            # TODO: Not checking for optimised conditions, these tend to vary a lot and I cannot be bothered to
            # add another parameter


        return differences

    return f

def generate_tests_from_options(options):

    if 'mea' in options.tests:
        for model in MODELS:
            for moment in range(2, options.max_moment+1):
                yield Test('MEA-{0}'.format(model),
                           MEA_TEMPLATE.format(model_file=os.path.join(options.inout_dir, model),
                                               moments=moment),
                           os.path.join(options.inout_dir, 'ODEout.tmp'),
                           os.path.join(options.model_answers_dir, 'MEA{0}'.format(moment), model + '.out'),
                           # todo use new comparison of equations:
                           #compare_ode_problems,
                           diff_comparison,
                           filter_function=filter_time_taken)

    if 'lna' in options.tests:
        for model in MODELS:
            yield Test('LNA-{0}'.format(model),
                       LNA_TEMPLATE.format(model_file=os.path.join(options.inout_dir, model)),
                       os.path.join(options.inout_dir, 'ODEout.tmp'),
                       os.path.join(options.model_answers_dir, 'LNA', model + '.out'),
                       diff_comparison,
                       filter_function=filter_time_taken)

    if 'simulation' in options.tests:
        if options.sundials_parameters is None:
            raise Exception("Cannot run simulation tests as no sundials parameters specified")

        for model in SIMULATION_MODELS:
            output_file = 'simout_{0}.txt'.format(model)
            yield Test('simulation-{0}-MEA'.format(model),
                       SIMULATION_TEMPLATE.format(model_file=os.path.join(options.inout_dir, 'model_{0}.txt'.format(model)),
                                                  sundials_parameters=options.sundials_parameters,
                                                  timeparam_file=os.path.join(options.inout_dir, 'param_{0}.txt'.format(model)),
                                                  output_file=output_file,
                                                  method='MEA'),
                       os.path.join(options.inout_dir, output_file),
                       os.path.join(options.model_answers_dir, 'sim', output_file),
                       compare_tsv_with_float_epsilon,
                       filter_function=filter_input_file)
            # Yeah: these won't be that easy to test, as they add some multivariate gaussian when simulating it, soz.
            # output_file_lna = 'simout_{0}_LNA.txt'.format(model)
            # yield Test('simulation-{0}-LNA'.format(model),
            #            SIMULATION_TEMPLATE.format(model_file=os.path.join(options.inout_dir, 'model_{0}.txt'.format(model)),
            #                                       sundials_parameters=options.sundials_parameters,
            #                                       timeparam_file=os.path.join(options.inout_dir, 'param_{0}.txt'.format(model)),
            #                                       output_file=output_file_lna,
            #                                       method='LNA'),
            #            os.path.join(options.inout_dir, output_file_lna),
            #            os.path.join(options.model_answers_dir, 'sim', output_file_lna),
            #            compare_tsv_with_float_epsilon,
            #            filter_function=filter_input_file)

    if 'inference' in options.tests:
        for model, dataset, model_answer in INFERENCE_MODELS:
            yield Test('inference-{0}-{1}'.format(model, dataset),
                       INFERENCE_TEMPLATE.format(model_file=os.path.join(options.inout_dir, 'model_{0}.txt'.format(model)),
                                                 sundials_parameters=options.sundials_parameters,
                                                 timeparam_file=os.path.join(options.inout_dir, 'param_{0}.txt'.format(model)),
                                                 dataset=dataset),
                       os.path.join(options.inout_dir, 'inferout.tmp'),
                       os.path.join(options.model_answers_dir, 'infer', model_answer),
                       parameter_and_distance_comparisons(),
                       filter_function=filter_input_file)
    if 'inference-with-restarts' in options.tests:
        for model, dataset, model_answer, allowed_slack, allowed_slack_params in INFERENCE_WITH_RESTARTS_MODELS:
            yield Test('inference-restarts-{0}-{1}'.format(model, dataset),
                       INFERENCE_WITH_RESTARTS_TEMPLATE.format(model_file=os.path.join(options.inout_dir, 'model_{0}.txt'.format(model)),
                                                               sundials_parameters=options.sundials_parameters,
                                                               timeparam_file=os.path.join(options.inout_dir, 'param_{0}.txt'.format(model)),
                                                               dataset=dataset),
                       os.path.join(options.inout_dir, 'inferout.restarts.tmp'),
                       os.path.join(options.model_answers_dir, 'infer', 'with-restarts', model_answer),
                       parameter_and_distance_comparisons(allowed_slack, allowed_slack_params),
                       filter_function=filter_input_file)

    if 'inference-with-distributions' in options.tests:
        for model, dataset, model_answer_template, allowed_slack, allowed_slack_params in INFERENCE_WITH_DISTRIBUTIONS_MODELS:
            for distribution in INFERENCE_DISTRIBUTIONS:
                yield Test('inference-restarts-{0}-{1}-{2}'.format(model, dataset, distribution),

                           INFERENCE_WITH_DISTRIBUTIONS_TEMPLATE.format(model_file=os.path.join(options.inout_dir, 'model_{0}.txt'.format(model)),
                                                                   sundials_parameters=options.sundials_parameters,
                                                                   timeparam_file=os.path.join(options.inout_dir, 'param_{0}.txt'.format(model)),
                                                                   dataset=dataset,
                                                                   distribution=distribution,
                                                                   restart_params=''),
                           os.path.join(options.inout_dir, 'inferout.tmp'),
                           os.path.join(options.model_answers_dir, 'infer', 'distributions', model_answer_template.format(distribution)),
                           parameter_and_distance_comparisons(),
                           filter_function=None)
                yield Test('inference-restarts-{0}-{1}-{2}'.format(model, dataset, distribution),

                           INFERENCE_WITH_DISTRIBUTIONS_TEMPLATE.format(model_file=os.path.join(options.inout_dir, 'model_{0}.txt'.format(model)),
                                                                   sundials_parameters=options.sundials_parameters,
                                                                   timeparam_file=os.path.join(options.inout_dir, 'param_{0}.txt'.format(model)),
                                                                   dataset=dataset,
                                                                   distribution=distribution,
                                                                   restart_params='--restart --nRestart=5 --random-seed=42'),
                           os.path.join(options.inout_dir, 'inferout.tmp'),
                           os.path.join(options.model_answers_dir, 'infer', 'distributions', 'with-restarts',
                                        model_answer_template.format(distribution)),
                           parameter_and_distance_comparisons(allowed_slack, allowed_slack_params),
                           filter_function=None)



def main():
    parser = create_options_parser()
    options = parser.parse_args()

    tests_to_run = list(generate_tests_from_options(options))
    number_of_tests = len(tests_to_run)

    if options.xunit:
        print '<testsuite tests="{0}">'.format(number_of_tests)


    for i, test in enumerate(tests_to_run):

        if not options.xunit:
            print '> Running test #{0}/{1} ({2})'.format(i+1, number_of_tests, test.name)
            print test

        exception = None
        traceback_ = None
        time_taken = None

        # Remove the previous output file
        test.cleanup()

        try:
            out, err, time_taken = test.run_command()
        except Exception, e:
            exception = e
            traceback_ = traceback.format_exc(10)

        differences = None
        if not exception:
            try:
                differences = test.compare_outputs()
            except Exception, e:
                exception = e
                traceback_ = traceback.format_exc(10)

        if options.xunit:
            print '<testcase classname="regression" name="{0}" time="{1}">'.format(test.name,
                                                                                         time_taken.total_seconds() if time_taken else "")
        if exception:
            if options.xunit:
                print '<failure type="Exception"><![CDATA[\n' \
                      'STDOUT:\n{out}------------\n' \
                       'STDERR:\n{err}------------\n' \
                       'TRACEBACK\n{traceback}]]></failure>'.format(out=out, err=err, traceback=traceback_)
                # Note that we need to execute all tests even if previous ones failed for xunit
            else:
                print '> Test Failed with exception {0!r}'.format(e)
                print traceback_
                break
        else:
            differences = test.compare_outputs()
            if not differences:
                if not options.xunit:
                    print "> ALL OK. Runtime: {0}s".format(time_taken.total_seconds())
                    print
            else:
                string_differences = '\n'.join(differences)
                if options.xunit:
                    print '<failure type="Output Mismatch"><![CDATA[\n' \
                          'STDOUT:\n{out}------------\n' \
                          'STDERR:\n{err}------------\n' \
                          'MISMATCH:\n{mismatch}]]></failure>'.format(out=out, err=err, mismatch=string_differences)
                    # Again no break here
                else:
                    print '> Test FAILED, here are the differences between files:'
                    print string_differences
                    break
        if options.xunit:
            print '</testcase>'

    if options.xunit:
        print '</testsuite>'

if __name__ == '__main__':
    main()
