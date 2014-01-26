import random
import numpy as np
#nSeg is number of segments/samples
#nDim is number of dimensions/variables/parameters
#y is a list of the ranges for each variable i.e. [(70,110),(0.1,0.5),...]
def hypercube(nSeg, y):
    """
    This implements Latin Hypercube Sampling.

    See https://mathieu.fenniak.net/latin-hypercube-sampling/ for intuitive explanation of what it is

    :param nSeg: number of segments/samples
    :param y: initial parameters and conditions (list of ranges, i.e. (70, 110), (0.1, 0.5) ..)
    :return:
    """

    nDim = len(y)

    # Split range 0-1 into `nSeg` segments of equal size
    possvalues = [0] * nSeg
    for i in range(nSeg):
        possvalues[i] = (((1.0) / nSeg) * i, ((1.0) / nSeg) * (i + 1))

    x = [0] * nDim
    for i in range(nDim):
        values = [0] * len(possvalues)
        for j in range(len(values)):
            # Set values[j] to a random value within the appropriate segment
            mult = random.random()
            values[j] = (mult * (possvalues[j][1] - possvalues[j][0])) + (possvalues[j][0])
        # I think this is equivalent to random.shuffle(values) except the latter does that in-place
        x[i] = random.sample(values, len(values))

    # at this point x is a list of lists containing a randomly-ordered list of random values
    # in each of the `possvalues` segments

    # values below should be replaced with possvalues or whole len() thing with `nSeg`
    samples = [0] * len(values)
    for i in range(len(values)):
        sample = [0] * nDim
        for j in range(nDim):
            sample[j] = x[j][i]
        samples[i] = sample
    # It looks like `samples` is just transposed version of `x`, i.e. `samples[i][j] = x[j][i]`

    for sample in samples:
        for i in range(len(sample)):
            # if no range given for parameter/variable
            if y[i][1] == y[i][0]:
                # just return the whatever constant was given
                sample[i] = y[i][1]
            else:
                # return the value indicated by random number in sample[i] that is within that range
                sample[i] = (sample[i] * (y[i][1] - y[i][0])) + y[i][0]

    return samples
