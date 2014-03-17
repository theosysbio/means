from means.core import Model

MODEL_MICHAELIS_MENTEN = Model(parameters=['c_0', 'c_1', 'c_2'],
                               species=['y_0', 'y_1'],
                               # TODO: 120-301 below are hardcoded because those are the `initial_values` for parameters
                               # for that model. I.e. these represent y0 and y1 at time zero, y0(0) and y1(0),
                               # we might want to handle this explicitly as when the initial parameters are different
                               # the model becomes less meaningful
                               propensities=['c_0*y_0*(120-301+y_0+y_1)',
                                             'c_1*(301-(y_0+y_1))',
                                             'c_2*(301-(y_0+y_1))'],
                               stoichiometry_matrix=[[-1, 1, 0],
                                                     [0, 0, 1]])

MODEL_DIMERISATION = Model(parameters=['c_0', 'c_1', 'c_2'],
                           species=['y_0'],
                           stoichiometry_matrix=[[-2, 2]],
                           propensities=['c_0*y_0*(y_0-1)',
                                         'c_1*((1.0/2)*(c_2-y_0))'])

MODEL_P53 = Model(parameters=['c_0',   # P53 production rate
                             'c_1',   # MDM2-independent p53 degradation rate
                             'c_2',   # saturating p53 degradation rate
                             'c_3',   # P53-dependent MDM2 production rate
                             'c_4',   # MDM2 maturation rate
                             'c_5',   # MDM2 degradation rate
                             'c_6'],  # P53 threshold of degradation by MDM2
                  species=['y_0',   # Concentration of p53
                           'y_1',   # Concentration of MDM2 precursor
                           'y_2'],  # Concentration of MDM2
                  stoichiometry_matrix=[[1, -1, -1, 0, 0, 0],
                                        [0, 0, 0, 1, -1, 0],
                                        [0, 0, 0, 0, 1, -1]],
                  propensities=['c_0',
                                'c_1*y_0',
                                'c_2*y_2*y_0/(y_0+c_6)',
                                'c_3*y_0',
                                'c_4*y_1',
                                'c_5*y_2'])


MODEL_HES1 = Model(parameters=['c_0', 'c_1', 'c_2', 'c_3'],
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

MODEL_LOTKA_VOLTERRA  =  Model(
                               parameters=['k_1', 'k_2', 'k_3'],
                               species=['Pred', 'Prey'],
                               propensities=['k_1 * Prey',
                                             'k_2 * Pred * Prey',
                                             'k_3 * Pred'],
                               stoichiometry_matrix=[[0, +1, -1],
                                                     [+1, -1, 0]])
