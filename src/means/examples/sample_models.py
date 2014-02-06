from means.model import Model

MODEL_MICHAELIS_MENTEN = Model(constants=['c_0', 'c_1', 'c_2'],
                               species=['y_0', 'y_1'],
                               propensities=['c_0*y_0*(120-301+y_0+y_1)',
                                             'c_1*(301-(y_0+y_1))',
                                             'c_2*(301-(y_0+y_1))'],
                               stoichiometry_matrix=[[-1, 1, 0], [0, 0, 1]])

MODEL_DIMERISATION = Model(constants=['c_0', 'c_1', 'c_2'],
                           species=['y_0'],
                           stoichiometry_matrix=[[-2, 2]],
                           propensities=['c_0*y_0*(y_0-1)',
                                         'c_1*((1.0/2)*(c_2-y_0))'])

MODEL_P53 = Model(constants=['c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6'],
                  species=['y_0', 'y_1', 'y_2'],
                  stoichiometry_matrix=[[1, -1, -1, 0, 0, 0],
                                        [0, 0, 0, 1, -1, 0],
                                        [0, 0, 0, 0, 1, -1]],
                  propensities=['c_0',
                                'c_1*y_0',
                                'c_2*y_2*y_0/(y_0+c_6)',
                                'c_3*y_0',
                                'c_4*y_1',
                                'c_5*y_2'])


MODEL_HES1 = Model(constants=['c_0', 'c_1', 'c_2', 'c_3'],
                   species=['y_0', 'y_1', 'y_2'],
                   propensities=['0.03*y_0',
                                 '0.03*y_1',
                                 '0.03*y_2',
                                 'c_3*y_1',
                                 'c_2*y_0',
                                 '1.0/(1+(y_2/c_0)**2)'],
                   stoichiometry_matrix=[[-1, 0, 0, 0, 0, 1],
                                         [0, -1, 0, -1, 1, 0],
                                         [0, 0, -1, 1, 0, 0]])



