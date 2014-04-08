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
<<<<<<< HEAD
    variable_parameters = ListOfKeyValuePairsParameter()

    xlim = Parameter(significant=False, default=None)
    ylim = Parameter(significant=False, default=None)
    vmax = Parameter(significant=False, default=None)
    vmin = Parameter(significant=False, default=None)

=======


>>>>>>> parent of bf62b7f... find pairs of parameters that generate overlapping optimal and observed trajectories
    def requires(self):
        return InferenceTOSSATask(model=self.model, max_order=self.max_order,closure=self.closure,
                                  multivariate=self.multivariate,
                                  parameters=self.parameters,
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
<<<<<<< HEAD
        variable_parameters = self.variable_parameters

=======
>>>>>>> parent of bf62b7f... find pairs of parameters that generate overlapping optimal and observed trajectories
        min_dist = min([x[2] for x in inference_result.distance_landscape])
        max_dist = max([x[2] for x in inference_result.distance_landscape])

        vmin = self.vmin if self.vmin is not None else min_dist
        vmax = self.vmax if self.vmax is not None else max_dist

        fig = plt.figure(figsize=(5,5), dpi=328)
        fig.subplots_adjust(wspace=0, hspace=0)

<<<<<<< HEAD
        parameters = [i for i,j in variable_parameters]
        dimension = len(parameters)

        if dimension > 2:
            for i,parameter_y in enumerate(parameters):
                for j, parameter_x in enumerate(parameters):

                    ax = plt.subplot(dimension,dimension, i*dimension + j + 1)
                    if i != j:
                        inference_result.plot_distance_landscape_projection(parameter_x, parameter_y,
                                                                            norm=LogNorm(), vmin=vmin, vmax=vmax,
                                                                            fmt='%.4g')

                    inference_result.plot_trajectory_projection(parameter_x,parameter_y,legend=False, ax=ax,
                                                                start_and_end_locations_only=i==j,
                                                                color='red',
                                                                start_marker='ro',
                                                                end_marker='rx')
                    if j != 0:
                        ax.yaxis.set_visible(False)

                    if i != dimension - 1:
                        ax.xaxis.set_visible(False)
        else:
            ax = plt.gca()
            parameter_x, parameter_y = parameters

            str_parameters = map(str, inference_result.problem.parameters)
            index_x = str_parameters.index(str(parameter_x))
            index_y = str_parameters.index(str(parameter_y))

            start_x = inference_result.starting_parameters[index_x]
            start_y = inference_result.starting_parameters[index_y]

            optimal_x = inference_result.optimal_parameters[index_x]
            optimal_y = inference_result.optimal_parameters[index_y]

            start_label = 'Start ($c_2={0:.4f}$, $c_6={1:.4f}$)'.format(start_x, start_y)
            end_label = 'End ($c_2={0:.4f}$, $c_6={1:.4f}$)'.format(optimal_x, optimal_y)

            inference_result.plot_distance_landscape_projection(parameter_x, parameter_y,
                                                                norm=LogNorm(), vmin=vmin, vmax=vmax,
                                                                fmt='%.4g')

            inference_result.plot_trajectory_projection(parameter_x, parameter_y,legend=False, ax=ax,
                                                        start_and_end_locations_only=False,
                                                        color='red',
                                                        start_marker='arrow',
                                                        end_marker='arrow', start_label=start_label,
                                                        end_label=end_label)
            xlim = ax.get_xlim()
=======
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
>>>>>>> parent of bf62b7f... find pairs of parameters that generate overlapping optimal and observed trajectories

            padding_x = (xlim[1] - xlim[0]) / 10.0
            xlim = (xlim[0]-padding_x, xlim[1]+padding_x)
            ax.set_xlim(xlim)

            ylim = ax.get_ylim()

            padding_y = (ylim[1] - ylim[0]) / 10.0
            ylim = (ylim[0]-padding_y, ylim[1]+padding_y)
            ax.set_ylim(ylim)


            if self.xlim is not None:
                ax.set_xlim(self.xlim)
            if self.ylim is not None:
                ax.set_ylim(self.ylim)


            ax.set_title('Max order = {0}'.format(self.max_order))

<<<<<<< HEAD
        return fig


class SampleMultidimensionInferenceFigure(MultiDimensionInferenceFigure):

    model = P53Model()
    #parameters = [90.0, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01]
=======
    parameters = [90.0, 0.002, 1.7, 1.1, 0.9, # todo: change to 0.93, thanx
                    0.96, 0.01]
>>>>>>> parent of bf62b7f... find pairs of parameters that generate overlapping optimal and observed trajectories
    initial_conditions = [70.0, 30.0, 60.0]
    timepoints_arange = [0.0, 40.0, 0.1]
    starting_initial_conditions = [70.0, 30.0, 60.0]
    n_simulations = IntParameter(default=5000)
    # Use a pair of parameters chosen based on the result from class FigureTwoParametersForInference
    variable_parameters = [('c_2', None), ('c_6', None)]

class MultiOrderMultiDimensionInferenceFigure(TexFigureTask):

    closure = MEATask.closure
    multivariate = MEATask.multivariate
    label = "MultiDimensional"
    caption = "MultiDimensional"
    max_order_list = ListParameter(default=[1,2,3,4,5])
    standalone=True
    number_of_columns = 1

    interesting_parameters = [
        [90.0, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01],
         [90.0, 0.002, 1.77, 1.1, 0.93, 0.96, 3.2736],
         [90.0, 0.002, 1.7040, 1.1, 0.93, 0.96, 0.7822],
    ]
    def requires(self):

        requirements = []
        for order in self.max_order_list:
            for p in self.interesting_parameters:
                figure = SampleMultidimensionInferenceFigure(max_order=order,
                                                             closure=self.closure,
                                                             multivariate=self.multivariate,
                                                             starting_parameters=p,
                                                             parameters=p,
                                                             xlim=(1.5, 2),
                                                             vmin=1,
                                                             vmax=1e10)

                requirements.append(figure)

                param_str = ', '.join(['{0:.4f}'.format(x) for x in p ])
                label = 'Inference starting at {0}, max order = {1}'.format(param_str, order)
                trajectory_figure = FigureInferenceStartEndSSA(model=SampleMultidimensionInferenceFigure.model,
                                           max_order=order,
                                           closure=self.closure,
                                           multivariate=self.multivariate,
                                           parameters=p,
                                           initial_conditions=SampleMultidimensionInferenceFigure.initial_conditions,
                                           timepoints_arange=SampleMultidimensionInferenceFigure.timepoints_arange,
                                           #solver=self.solver, solver_kwargs=self.solver_kwargs,
                                           starting_parameters=p,
                                  starting_initial_conditions=SampleMultidimensionInferenceFigure.initial_conditions,
                                  variable_parameters=SampleMultidimensionInferenceFigure.variable_parameters,
                                  # distance_function_type=self.distance_function_type,
                                  n_simulations=5000, # TODO: Find where this comes from
                                  label=label
                                  )

                requirements.append(trajectory_figure)

        return requirements




<<<<<<< HEAD

=======
    starting_parameters = [90.0, 0.002, 1.7, 1.1, 0.9, # todo: change to 0.93, thanx
                    0.96, 0.01]
    starting_initial_conditions = [70.0, 30.0, 60.0]

    variable_parameters = zip(P53Model().parameters, [None] * len(P53Model().parameters))
>>>>>>> parent of bf62b7f... find pairs of parameters that generate overlapping optimal and observed trajectories



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

            inference = FigureInferenceStartEndSSA(model=self.model, max_order=self.max_order,closure=self.closure,
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

class FigureInferenceStartEndSSA(FigureTask):

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
        opt, start, obs = None, None, None
        for starting, optimal in zip(result.starting_trajectories, result.optimal_trajectories):
            assert(starting.description == optimal.description)
            try:
                observed_trajectory = observed_trajectories_lookup[optimal.description]
            except KeyError:
                continue
            sum_of_sqr_distance = np.sum(np.square(observed_trajectory.values - optimal.values))

            subplot_number += 1
            ax = plt.subplot(1, n_columns, subplot_number)
            ax.annotate('Distance={0:.2f}'.format(sum_of_sqr_distance), xy=(1, 0), xycoords='axes fraction', fontsize=16,
                xytext=(-5, 5), textcoords='offset points',
                ha='right', va='bottom')
            plt.subplot(1,n_columns, subplot_number)
            plt.title(observed_trajectory.description.mathtext())
            observed_trajectory.plot(marker='x',color='k', label='SSA', linestyle='None')
            optimal.plot(color='b', label='Optimal')
            starting.plot(color='r', label='Starting')

        plt.legend()
        plt.suptitle(self.label)

        return fig



if __name__ == '__main__':
    #run(main_task_cls=FigureHitAndMissTex)
    run()