"""
Simulates data for given model, moments, parameters, initial conditions
and method (moment expansion or LNA)
"""
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import sympy
from means.simulation.trajectory import Trajectory
from means.simulation.solvers import NP_FLOATING_POINT_PRECISION, CVodeSolver

# These are the default values in solver.c but they seem very low
from means.approximation.ode_problem import Moment, Descriptor, VarianceTerm


class SensitivityTerm(Descriptor):
    r"""
    A :class:`~means.approximation.ode_problem.Descriptor` term that describes a particular object represents the sensitivity
    of some ODE term with respect to some parameter.
    In other words, sensitivity term describes :math:`s_{ij}(t) = \frac{\partial y_i(t)}{\partial p_j}` where
    :math:`y_i` is the ODE term described above and :math:`p_j` is the parameter.

    This class is used to describe sensitivity trajectories returned by :class:`means.simulation.simulate.Simulation`
    """
    _ode_term = None
    _parameter = None

    def __init__(self, ode_term, parameter):
        """

        :param ode_term: the ode term whose sensitivity is being computed
        :type ode_term: :class:`~means.approximation.ode_problem.ODETermBase`
        :param parameter: parameter w.r.t. which the sensitivity is computed
        :type parameter: :class:`sympy.Symbol`
        """
        self._ode_term = ode_term
        self._parameter = parameter

    @property
    def ode_term(self):
        return self._ode_term

    @property
    def parameter(self):
        return self._parameter

    def __repr__(self):
        return '<Sensitivity of {0!r} w.r.t. {1!r}>'.format(self.ode_term, self.parameter)

    def __mathtext__(self):
        # Double {{ and }} in multiple places as to escape the curly braces in \frac{} from .format
        return r'$\frac{{\partial {0}}}{{\partial {1}}}$'.format(self.ode_term.symbol, self.parameter)

def validate_problem(problem):

    problem.validate()

    if problem.method == "MEA":
        moments = filter(lambda x: isinstance(x, Moment), problem.ordered_descriptions)
        if problem.left_hand_side.rows != len(moments):
            raise ValueError("There are {0} equations and {1} moments. "
                             "For MEA problems, the same number is expected.".format(problem.left_hand_side.rows,
                                                                                     len(moments)))
    elif problem.method == 'LNA':
        # FIXME: do some validation for LNA here
        pass



class Simulation(object):
    """
    An object that provides wrappers around CVode library to allow simulation of the ODE systems.
    """
    __problem = None
    _postprocessing = None
    _solver_options = None
    _solver_short_name = None

    def __init__(self, problem, solver_short_name='cvode', **solver_options):
        """

        :param problem:
        :type problem: ODEProblem
        :param compute_sensitivities: Whether the model should test parameter sensitivity or not
        :param solver_short_name: the solver to use TODO: list available solvers
        :param solver_options: options to set in the solver
        """
        self.__problem = problem
        validate_problem(problem)

        if problem.method == 'LNA':
            self._postprocessing = _postprocess_lna_simulation
        else:
            self._postprocessing = _postprocess_default

        self._solver_short_name = solver_short_name
        self._solver_options = solver_options

    def simulate_system(self, parameters, initial_conditions, timepoints):
        """
        Simulates the system for each of the timepoints, starting at initial_constants and initial_values values

        :param parameters: list of the initial values for the constants in the model.
                                  Must be in the same order as in the model
        :param initial_conditions: List of the initial values for the equations in the problem. Must be in the same order as
                               these equations occur.
                               If not all values specified, the remaining ones will be assumed to be 0.
        :param timepoints: A list of time points to simulate the system for
        :return: a list of :class:`~means.simulation.simulate.Trajectory` objects,
                 one for each of the equations in the problem
        """

        # If not all intial conditions specified, append zeros to them
        # TODO: is this really the best way to do this?
        if len(initial_conditions) < self.problem.number_of_equations:
            initial_conditions = np.concatenate((initial_conditions,
                                                 [0.0] * (self.problem.number_of_equations - len(initial_conditions))))

        solver = CVodeSolver(self.problem, parameters, initial_conditions, starting_time=timepoints[0],
                             **self._solver_options)
        trajectories = solver.simulate(timepoints)

        return self._postprocessing(self.problem, trajectories)

    @property
    def problem(self):
        return self.__problem

class SimulationWithSensitivities(Simulation):
    """
    A similar object to it's baseclass :class:`~means.simulation.simulate.Simulation`, however provides
    instances of :class:`~means.simulation.simulate.TrajectoryWithSensitivityData` objects as a result instead.
    """

    def __init__(self, problem, **solver_options):
        # Hardcode CVODE solver for sensitivity simulations
        super(SimulationWithSensitivities, self).__init__(problem, solver_short_name='cvode', **solver_options)


    def _initialise_solver(self, initial_constants, initial_values, initial_timepoint=0.0):
        solver = super(SimulationWithSensitivities, self)._initialise_solver(initial_constants, initial_values,
                                                                             initial_timepoint)
        # Override sensitivity settings of solver
        solver.usesens = True
        solver.report_continuously = True

        return solver

    def _simulate(self, solver, initial_constants, initial_values, timepoints):
        trajectories = super(SimulationWithSensitivities, self)._simulate(solver, initial_constants, initial_values,
                                                                          timepoints)

        sensitivities_raw = np.array(solver.p_sol)

        sensitivity_values = []
        for i, ode_term in enumerate(self.problem.ode_lhs_terms):
            term_sensitivities = []
            for j, parameter in enumerate(self.problem.constants):
                term_sensitivities.append((parameter, sensitivities_raw[j, :, i]))
            sensitivity_values.append(term_sensitivities)

        trajectories_with_sensitivity_data = []
        for trajectory, sensitivities in zip(trajectories, sensitivity_values):

            # Collect the sensitivities into a nice dictionary of Trajectory objects
            sensitivity_trajectories = []
            for parameter, values in sensitivities:
                sensitivity_trajectories.append(Trajectory(trajectory.timepoints, values,
                                                           SensitivityTerm(trajectory.description, parameter)))

            trajectory_with_sensitivities = TrajectoryWithSensitivityData.from_trajectory(trajectory,
                                                                                          sensitivity_trajectories)
            trajectories_with_sensitivity_data.append(trajectory_with_sensitivities)

        return trajectories_with_sensitivity_data

    def simulate_system(self, parameters, initial_conditions, timepoints):
        """
        Simulates the system for each of the timepoints, starting at initial_constants and initial_values values

        :param parameters: list of the initial values for the constants in the model.
                                  Must be in the same order as in the model
        :param initial_conditions: List of the initial values for the equations in the problem. Must be in the same order as
                               these equations occur.
                               If not all values specified, the remaining ones will be assumed to be 0.
        :param timepoints: A list of time points to simulate the system for
        :return: a list of :class:`~means.simulation.simulate.TrajectoryWithSensitivityData` objects,
                 one for each of the equations in the problem
        """
        return super(SimulationWithSensitivities, self).simulate_system(parameters, initial_conditions, timepoints)


def _postprocess_default(problem, trajectories):
    return trajectories

def _postprocess_lna_simulation(problem, trajectories):
    timepoints = trajectories[0].timepoints

    # TODO: this should be going through the descriptions of LHS fields, rather than number of species
    # Would make code cleaner

    number_of_species = problem.number_of_species

    sampled_observations = np.zeros((len(timepoints), number_of_species), dtype=NP_FLOATING_POINT_PRECISION)
    variance_trajectories = filter(lambda x: isinstance(x.description, VarianceTerm), trajectories)
    species_trajectories = filter(lambda x: not isinstance(x.description, VarianceTerm), trajectories)

    for t in range(len(timepoints)):
        covariance_matrix = sympy.Matrix(np.zeros((number_of_species, number_of_species)))

        for trajectory in variance_trajectories:
            values = trajectory.values

            if not isinstance(trajectory.description, VarianceTerm):
                continue
            i, j = trajectory.description.position
             # FIXME: hack to make regression tests still work, remove
            if i*number_of_species+j >= 2*number_of_species:
                continue
            covariance_matrix[i, j] = values[t]

        means = [ trajectory.values[t] for trajectory in species_trajectories ]

        # Recreate the species trajectories by sampling from multivariate normal
        sampled_observations[t] = np.random.multivariate_normal(means, covariance_matrix)

    # Recompile everything back to trajectories
    answer = []

    for i, old_trajectory in enumerate(species_trajectories):
        new_trajectory = Trajectory(old_trajectory.timepoints, sampled_observations[:, i], old_trajectory.description)
        answer.append(new_trajectory)

    return answer


def print_output(output_file, trajectories, initial_conditions, number_of_species, param, timepoints, max_order=None):
    # Check maximum order of moments to output to file/plot
    # TODO: change wherever maxorder comes from for it to be "None" not false.

    # write results to output file (Input file name, parameters,
    # initial conditions, data needed for maximum entropy
    # (no.timepoints, no.species, max order of moments),
    # timepoints, and trajectories for each moment)
    output = open(output_file, 'w')
    try:
        output.write('\n>Parameters: {0!r}\n>Starting values: {1}\n'.format([round(x, 6) for x in param],
                                                                            [round(y, 6) for y in initial_conditions]))

        output.write('#\t{0}\t{1}\t{2}\n'.format(len(timepoints), number_of_species, max_order))
        output.write('time\t{0}\n'.format('\t'.join(map(str, timepoints))))

        # write trajectories of moments (up to maxorder) to output file
        for trajectory in trajectories:
            term = trajectory.description
            if not isinstance(term, Moment):
                continue
            if max_order is None or term.order <= max_order:
                output.write('{0}\t{1}\n'.format(term, '\t'.join(map(str, trajectory.values))))
    finally:
        output.close()

def simulate(problem, trajout, timepoints, initial_constants, initial_variables, maxorder):
    """
    :param simulation_type: either "MEA" or "LNA"
    :param problem: Parsed problem to simulate
    :type problem: ODEProblem
    :param trajout: Name of output file for this function (where simulated trajectories would be stored, i.e. --simout)
    :param timepoints: List of timepoints
    :param initial_constants: List of kinetic parameters
    :param initial_variables: List of initial conditions for each moment (in timeparameters file)
    :param maxorder: Maximum order of moments to output to either plot or datafile. (Defaults to maximum order of moments)
    :return: a list of trajectories resulting from simulation
    """

    # Get required info from the expansion output

    number_of_species = problem.number_of_species

    term_descriptions = problem.ordered_descriptions

    initial_variables = np.array(initial_variables, dtype=NP_FLOATING_POINT_PRECISION)
    initial_constants = np.array(initial_constants, dtype=NP_FLOATING_POINT_PRECISION)
    simulator = Simulation(problem)
    trajectories = simulator.simulate_system(initial_constants, initial_variables, timepoints)

    print_output(trajout, trajectories, initial_variables, number_of_species,
                 initial_constants, timepoints, maxorder)

    return trajectories, term_descriptions

def graphbuilder(soln,momexpout,title,t,momlist):
    """
    Creates a plot of the solutions

    :param soln: output from CVODE (array of solutions at each time point for each moment)
    :param momexpout:
    :param title:
    :param t:
    :param momlist:
    :return:
    """
    simtype = 'momexp'
    LHSfile = open(momexpout)
    lines = LHSfile.readlines()
    LHSindex = lines.index('LHS:\n')
    cindex = lines.index('Constants:\n')
    LHS = []
    for i in range(LHSindex+1,cindex-1):
        LHS.append(lines[i].strip())
        if lines[i].startswith('V'):simtype='LNA'
    fig = plt.figure()
    count = -1
    for el in LHS:
        if '_' in el:
            count+=1
    n = np.floor(np.sqrt(len(LHS)+1-count))+1
    m = np.floor((len(LHS)+1-count)/n)+1
    if n>3:
        n=3
        m=3
    for i in range(len(LHS)):
        if simtype == 'LNA':
            if 'y' in LHS[i]:
                ax = plt.subplot(111)
                ax.plot(t,soln[:][i],label=LHS[i])
                plt.xlabel('Time')
                plt.ylabel('Means')
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          fancybox=True, shadow=True, ncol=3)
        elif simtype == 'momexp':
            if i<(count+9):
                if '_' in LHS[i]:
                    ax1 = plt.subplot(n,m,1)
                    ax1.plot(t,soln[:,i],label=LHS[i])
                    plt.xlabel('Time')
                    plt.ylabel('Means')
                    ax1.legend(loc='upper center', bbox_to_anchor=(0.9, 1.0),
          fancybox=True, shadow=True,prop={'size':12})
                else:
                    ax = plt.subplot(n,m,i+1-count)
                    ax.plot(t,soln[:,i])
                    plt.xlabel('Time')
                    plt.ylabel('['+str(momlist[i])+']')


    fig.suptitle(title)
    plt.show()

