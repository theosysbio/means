from __future__ import absolute_import, print_function

import random

import numpy as np

def hypercube(number_of_samples, variables):
    """
    This implements Latin Hypercube Sampling.

    See https://mathieu.fenniak.net/latin-hypercube-sampling/ for intuitive explanation of what it is

    :param number_of_samples: number of segments/samples
    :param variables: initial parameters and conditions (list of ranges, i.e. (70, 110), (0.1, 0.5) ..)
    :return:
    """
    number_of_dimensions = len(variables)

    # Split range 0-1 into `nSeg` segments of equal size
    segment_ranges = []
    for i in range(number_of_samples):
        ratio = 1.0 / number_of_samples
        segment_ranges.append((ratio * i, ratio * (i + 1)))

    x = []
    for i in range(number_of_dimensions):
        values = []
        for j, segment in enumerate(segment_ranges):
            # Set values[j] to a random value within the appropriate segment
            random_element = random.random()
            value = (random_element * (segment[1] - segment[0])) + (segment[0])
            values.append(value)

        # TODO: replace the below line with random.shuffle(values) (no need values= in front)
        # this breaks regression tests as the values are shuffled in different order
        values = random.sample(values, len(values))
        x.append(values)

    # at this point x is a list of lists containing a randomly-ordered list of random values
    # in each of the `possvalues` segments

    samples = []
    for i in range(len(segment_ranges)):
        sample = [y[i] for y in x]
        samples.append(sample)
    # It looks like `samples` is just transposed version of `x`, i.e. `samples[i][j] = x[j][i]`

    for sample in samples:
        for i, variable in enumerate(variables):
            # if no range given for parameter/variable
            if variable[1] == variable[0]:
                # just return the whatever constant was given
                sample[i] = variable[1]
            else:
                # return the value indicated by random number in sample[i] that is within that range
                sample[i] = (sample[i] * (variable[1] - variable[0])) + variable[0]

    return samples
