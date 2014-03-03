import cPickle as pickle
from collections import OrderedDict
import pandas as pd
import os
from worker import MODEL

def compile_data_from_dir(dirname):

    df = []
    for data_filename in os.listdir(dirname):
        data_filename = os.path.join(dirname, data_filename)
        with open(data_filename, 'r') as data_file:
            data = pickle.load(data_file)

        kwargs = data['kwargs']
        parameters = kwargs.pop('parameters')
        initial_conditions = kwargs.pop('initial_conditions')

        row = OrderedDict()
        for symbol, parameter in zip(MODEL.constants, parameters):
            row[symbol] = parameter

        for symbol, initial_condition in zip(MODEL.species, initial_conditions):
            row[symbol] = initial_condition
        exception = data['exception']
        row['exception'] = exception
        for key, value in kwargs.iteritems():
            row[key] = value

        df.append(row)

    return pd.DataFrame(df)

if __name__ == '__main__':
    DATA = compile_data_from_dir('.data')


