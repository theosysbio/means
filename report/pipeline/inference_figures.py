from means import TrajectoryCollection, SolverException
from means.pipes import *
import numpy as np


from means.util.logs import get_logger
logger = get_logger(__name__)

class P53Model(means.Model):
    """
    A wrapper around means. Model that initialises it with P53 parameters and changes the __unicode__ function to
    print a shorter string
    """
    def __init__(self):
        from means.examples import MODEL_P53
        super(P53Model, self).__init__(MODEL_P53.species, MODEL_P53.parameters, MODEL_P53.propensities,
                                       MODEL_P53.stoichiometry_matrix)

    def __str__(self):
        # Override the str() methods so they do not print the whole blerch of things, but
        # only a nice and easily readable "p53"
        return 'p53'

class InferenceTOSSATask(InferenceTask):


    # General parameters for trajectory
    parameters = ListParameter(item_type=float)
    """Parameters to simulate trajectories for"""

    initial_conditions = ListParameter(item_type=float)
    """Initial conditions to use"""

    timepoints_arange = ListParameter(item_type=float)
    """An arangement of the timepoints to simulate,
       e.g. ``(0, 40, 0.1)`` would simulate from 0 to 40 in 0.1 increments"""

    n_simulations = IntParameter()

    observed_trajectories = None

    def requires(self):
        parents_require = super(InferenceTOSSATask,self).requires()
        ssa_task = SSATrajectoryTask(model=self.model,parameters=self.parameters,
                                     initial_conditions=self.initial_conditions,
                                        timepoints_arange=self.timepoints_arange,
                                        n_simulations=self.n_simulations)
        return [parents_require, ssa_task]

    def _return_object(self):
        problem_buffer, ssa_buffer = self.input()

        problem = problem_buffer.load()
        ssa_trajectories = ssa_buffer.load()

        self.observed_trajectories = ssa_trajectories
        inference_result = self._compute_inference_result(problem)

        return inference_result

class MultiDimensionInferenceFigure(FigureTask):

    model = ModelParameter()
    max_order = MEATask.max_order
    closure = MEATask.closure
    multivariate = MEATask.multivariate
    parameters = ListParameter(item_type=float)
    initial_conditions = ListParameter(item_type=float)
    timepoints_arange = ListParameter(item_type=float)
    solver = luigi.Parameter(default='ode15s')
    solver_kwargs = ListOfKeyValuePairsParameter(default=[])
    starting_parameters = ListParameter(item_type=float)
    starting_initial_conditions = ListParameter(item_type=float)
    variable_parameters = ListOfKeyValuePairsParameter()
    distance_function_type = luigi.Parameter(default='sum_of_squares')
    n_simulations = IntParameter()


    def requires(self):
        return InferenceTOSSATask(model=self.model, max_order=self.max_order,closure=self.closure,
                                  multivariate=self.multivariate, parameters=self.parameters,
                                  initial_conditions=self.initial_conditions, timepoints_arange=self.timepoints_arange,
                                  solver=self.solver, solver_kwargs=self.solver_kwargs,
                                  starting_parameters=self.starting_parameters,
                                  starting_initial_conditions=self.starting_initial_conditions,
                                  variable_parameters=self.variable_parameters,
                                  distance_function_type=self.distance_function_type,
                                  n_simulations=self.n_simulations,
                                  return_distance_landscape=True,
                                  return_intermediate_solutions=True)

    def _return_object(self):
        from matplotlib import pyplot as plt
        from matplotlib.colors import LogNorm

        inference_result = self.input().load()
        min_dist = min([x[2] for x in inference_result.distance_landscape])
        max_dist = max([x[2] for x in inference_result.distance_landscape])

        fig = plt.figure(figsize=(20,20), dpi=328)
        fig.subplots_adjust(wspace=0, hspace=0)

        dimension = len(inference_result.problem.parameters)
        for i,parameter_y in enumerate(inference_result.problem.parameters):
            for j, parameter_x in enumerate(inference_result.problem.parameters):

                ax = plt.subplot(dimension,dimension, i*dimension + j + 1)
                if i != j:
                    inference_result.plot_distance_landscape_projection(parameter_x, parameter_y,
                                                                        norm=LogNorm(), vmin=min_dist, vmax=max_dist)

                inference_result.plot_trajectory_projection(parameter_x,parameter_y,legend=False, ax=ax,
                                                            start_and_end_locations_only=i==j,
                                                            color='red',
                                                            start_marker='ro',
                                                            end_marker='rx')
                if j != 0:
                    ax.yaxis.set_visible(False)

                if i != dimension - 1:
                    ax.xaxis.set_visible(False)

        return fig



class SampleMultidimensionInferenceFigure(MultiDimensionInferenceFigure):

    model = P53Model()

    #max_order = MEATask.max_order
    #closure = MEATask.closure
    #multivariate = MEATask.multivariate

    parameters = [90.0, 0.002, 1.7, 1.1, 0.9, # todo: change to 0.93, thanx
                    0.96, 0.01]
    initial_conditions = [70.0, 30.0, 60.0]
    timepoints_arange = [0.0, 40.0, 0.1]

    #solver = luigi.Parameter(default='ode15s')
    #solver_kwargs = ListOfKeyValuePairsParameter(default=[])

    starting_parameters = [90.0, 0.002, 1.7, 1.1, 0.9, # todo: change to 0.93, thanx
                    0.96, 0.01]
    starting_initial_conditions = [70.0, 30.0, 60.0]

    variable_parameters = zip(P53Model().parameters, [None] * len(P53Model().parameters))

    #distance_function_type = luigi.Parameter(default='sum_of_squares')
    n_simulations = IntParameter(default=5000)


class FindTwoParametersForInference(Task):

    model = P53Model()

    max_order = MEATask.max_order
    closure = MEATask.closure
    multivariate = MEATask.multivariate

    parameters = [90.0, 0.002, 1.7, 1.1, 0.93, # todo: change to 0.93, thanx
                    0.96, 0.01]
    initial_conditions = [70.0, 30.0, 60.0]
    timepoints_arange = [0.0, 40.0, 0.1]

    solver = luigi.Parameter(default='ode15s')
    solver_kwargs = ListOfKeyValuePairsParameter(default=[])

    starting_parameters = [90.0, 0.002, 1.7, 1.1, 0.93, # todo: change to 0.93, thanx
                    0.96, 0.01]
    starting_initial_conditions = [70.0, 30.0, 60.0]

    distance_function_type = luigi.Parameter(default='sum_of_squares')
    n_simulations = IntParameter(default=5000)


    def requires(self):
        requirements = []
        from itertools import combinations

        for v1,v2 in combinations(self.model.parameters, 2):
            variable_parameters = {(v1, None),(v2, None)}

            inference = FigureTwoParameterForInference(model=self.model, max_order=self.max_order,closure=self.closure,
                                  multivariate=self.multivariate, parameters=self.parameters,
                                  initial_conditions=self.initial_conditions, timepoints_arange=self.timepoints_arange,
                                  solver=self.solver, solver_kwargs=self.solver_kwargs,
                                  starting_parameters=self.starting_parameters,
                                  starting_initial_conditions=self.starting_initial_conditions,
                                  variable_parameters=variable_parameters,
                                  distance_function_type=self.distance_function_type,
                                  n_simulations=self.n_simulations,
                                    label='${0}$ and ${1}$'.format(v1,v2))
            requirements.append(inference)
        return requirements

    def _return_object(self):
        return 'running'

class FigureTwoParameterForInference(FigureTask):

    model = ModelParameter()
    max_order = MEATask.max_order
    closure = MEATask.closure
    multivariate = MEATask.multivariate
    parameters = ListParameter(item_type=float)
    initial_conditions = ListParameter(item_type=float)
    timepoints_arange = ListParameter(item_type=float)
    solver = luigi.Parameter(default='ode15s')
    solver_kwargs = ListOfKeyValuePairsParameter(default=[])
    starting_parameters = ListParameter(item_type=float)
    starting_initial_conditions = ListParameter(item_type=float)
    variable_parameters = ListOfKeyValuePairsParameter()
    distance_function_type = luigi.Parameter(default='sum_of_squares')
    n_simulations = IntParameter()
    label = Parameter(default=None, significant=False)

    def requires(self):
        return InferenceTOSSATask(model=self.model, max_order=self.max_order,closure=self.closure,
                                  multivariate=self.multivariate, parameters=self.parameters,
                                  initial_conditions=self.initial_conditions, timepoints_arange=self.timepoints_arange,
                                  solver=self.solver, solver_kwargs=self.solver_kwargs,
                                  starting_parameters=self.starting_parameters,
                                  starting_initial_conditions=self.starting_initial_conditions,
                                  variable_parameters=self.variable_parameters,
                                  distance_function_type=self.distance_function_type,
                                  n_simulations=self.n_simulations,
                                  return_distance_landscape=True,
                                  return_intermediate_solutions=True)

    def _return_object(self):
        from matplotlib import pyplot as plt
        result = self.input().load()
        fig = plt.figure(figsize=(15,5),dpi=327)
        observed_trajectories_lookup = {obs_traj.description: obs_traj for obs_traj in result.observed_trajectories}
        subplot_number = 0
        n_columns = len(result.observed_trajectories)
        for optimal in result.optimal_trajectories:
            try:
                observed_trajectory = observed_trajectories_lookup[optimal.description]
            except KeyError:
                continue

            subplot_number += 1
            plt.subplot(1,n_columns, subplot_number)
            plt.title(observed_trajectory.description.mathtext())
            observed_trajectory.plot(marker='x',color='k', label='Observed',linestyle='None')
            optimal.plot(color='b',label='Optimal')
        plt.legend()
        plt.suptitle(self.label)

        return fig







if __name__ == '__main__':
    #run(main_task_cls=FigureHitAndMissTex)
    run()