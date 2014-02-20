import numpy as np
from means.approximation.ode_problem import Descriptor


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

class Trajectory(object):
    """
    A single simulated or observed trajectory for an ODE term.
    """
    _timepoints = None
    _values = None
    _description = None

    def __init__(self, timepoints, values, description):
        """

        :param timepoints: timepoints the trajectory was simulated for
        :type timepoints: :class:`iterable`
        :param values: values of the curve at each of the timepoints
        :type values: :class:`iterable`
        :param description: description of the trajectory
        :type description: :class:`~means.approximation.ode_problem.Descriptor`
        """
        self._timepoints = np.array(timepoints)
        self._values = np.array(values)
        self._description = description

        assert(self._timepoints.shape == self._values.shape)

    @property
    def timepoints(self):
        """
        The timepoints trajectory was simulated for.

        :rtype: :class:`numpy.ndarray`
        """
        return self._timepoints

    @property
    def values(self):
        """
        The values for each of the timepoints in :attr:`~Trajectory.timepoints`.

        :rtype: :class:`numpy.ndarray`
        """
        return self._values

    @property
    def description(self):
        """
        Description of this trajectory. The same description as the description for particular ODE term.

        :rtype: :class:`~means.approximation.ode_problem.ODETermBase`
        """
        return self._description

    def plot(self, *args, **kwargs):
        """
        Plots the trajectory using :mod:`matplotlib.pyplot`.

        :param args: arguments to pass to :func:`~matplotlib.pyplot.plot`
        :param kwargs: keyword arguments to pass to :func:`~matplotlib.pyplot.plot`
        :return: the result of the :func:`matplotlib.pyplot.plot` function.
        """
        from means.plotting.util import mathtextify
        from matplotlib import pyplot as plt
        # Get label from the kwargs provided, or use self.description as default
        label = kwargs.pop('label', mathtextify(self.description))
        # This is needed for matplotlib version 1.1.1
        label = str(label)
        return plt.plot(self.timepoints, self.values, *args, label=label, **kwargs)

    def __repr__(self):
        return '{0}({1}, {2}, {3})'.format(self.__class__.__name__, self.timepoints, self.values, self.description)

    def __eq__(self, other):
        return np.equal(self.timepoints, other.timepoints).all() and np.equal(self.values, other.values).all() \
            and self.description == other.description

class TrajectoryWithSensitivityData(Trajectory):
    """
    An extension to :class:`~means.simulation.simulate.Trajectory` that provides data about the sensitivity
    of said trajectory as well.

    """

    _sensitivity_data = None

    def __init__(self, timepoints, values, description, sensitivity_data):
        """

        :param timepoints: timepoints the trajectory was simulated for
        :type timepoints: :class:`numpy.ndarray`
        :param values: values of the curve at each of the timepoints
        :type values: :class:`numpy.ndarray`
        :param description: description of the trajectory
        :type description: :class:`~means.approximation.ode_problem.Descriptor`
        :param sensitivity_data: a list of :class:`~means.approximation.simulation.simulate.Trajectory` objects
                                 signifying the sensitivity change over time for each of the parameters.
        :type sensitivity_data: list[:class:`~means.approximation.simulation.simulate.Trajectory`]
        """
        super(TrajectoryWithSensitivityData, self).__init__(timepoints, values, description)
        self._sensitivity_data = sensitivity_data

    @classmethod
    def from_trajectory(cls, trajectory, sensitivity_data):
        return cls(trajectory.timepoints, trajectory.values, trajectory.description, sensitivity_data)

    @property
    def sensitivity_data(self):
        """
        THe sensitivity data for the trajectory

        :rtype: list[:class:`~means.approximation.simulation.simulate.Trajectory`]
        """
        return self._sensitivity_data

    def __eq__(self, other):
        return isinstance(other, TrajectoryWithSensitivityData) and \
               super(TrajectoryWithSensitivityData, self).__eq__(other) and \
               self.sensitivity_data == other.sensitivity_data

    def plot_perturbations(self, parameter, delta=1e-4, *args, **kwargs):
        from matplotlib import pyplot as plt
        alpha = kwargs.pop('alpha', 0.1)
        for sensitivity_trajectory in self.sensitivity_data:
            description_parameter = sensitivity_trajectory.description.parameter
            # Compare them as strings so it is easier to pass it in
            if str(sensitivity_trajectory.description.parameter) != str(parameter):
                continue

            perturbed_trajectory_positive = perturbed_trajectory(self, sensitivity_trajectory, delta=delta)
            perturbed_trajectory_negative = perturbed_trajectory(self, sensitivity_trajectory, delta=-delta)
            plt.fill_between(self.timepoints,
                             perturbed_trajectory_negative.values,
                             perturbed_trajectory_positive.values,
                             alpha=alpha,
                             *args,
                             **kwargs)

            label = kwargs.pop('label', "${0}$, "
                                        "when ${1}$ is perturbed by ${2}$".format(self.description.symbol,
                                                                                  sensitivity_trajectory.description.parameter,
                                                                                  delta))
            # Fill_between does not generate a legend entry, use this hack with Rectangle to do this
            plt.gca().add_patch(plt.Rectangle((0, 0), 0, 0, alpha=alpha,
                                              label=label,
                                                    *args, **kwargs))


class PerturbedTerm(Descriptor):
    r"""
    A :class:`~means.approximation.ode_problem.Descriptor` term that describes a particular object represents the sensitivity
    of some ODE term with respect to some parameter.
    In other words, sensitivity term describes :math:`s_{ij}(t) = \frac{\partial y_i(t)}{\partial p_j}` where
    :math:`y_i` is the ODE term described above and :math:`p_j` is the parameter.

    This class is used to describe sensitivity trajectories returned by :class:`means.simulation.simulate.Simulation`
    """
    _ode_term = None
    _parameter = None
    _delta = None

    def __init__(self, ode_term, parameter, delta=0.01):
        """

        :param ode_term: the ode term whose sensitivity is being computed
        :type ode_term: :class:`~means.approximation.ode_problem.ODETermBase`
        :param parameter: parameter w.r.t. which the sensitivity is computed
        :type parameter: :class:`sympy.Symbol`
        """
        self._ode_term = ode_term
        self._parameter = parameter
        self._delta = delta

    @property
    def ode_term(self):
        return self._ode_term

    @property
    def parameter(self):
        return self._parameter

    @property
    def delta(self):
        return self._delta

    def __repr__(self):
        return '<Perturbed {0!r} when {1!r} is perturbed by {2!r}>'.format(self.ode_term, self.parameter, self.delta)

    def __mathtext__(self):
        # Double {{ and }} in multiple places as to escape the curly braces in \frac{} from .format
        return r'${0}$ when ${1}={1}+{2}$'.format(self.ode_term.symbol, self.parameter, self.delta)


def perturbed_trajectory(trajectory, sensitivity_trajectory, delta=1e-4):
    """
    Slightly perturb trajectory wrt the parameter specified in sensitivity_trajectory.

    :param trajectory: the actual trajectory for an ODE term
    :type trajectory: :class:`Trajectory`
    :param sensitivity_trajectory: sensitivity trajectory (dy/dpi for all timepoints t)
    :type sensitivity_trajectory: :class:`Trajectory`
    :param delta: the perturbation size
    :type delta: float
    :return: :class:`Trajectory`
    """
    sensitivity_trajectory_description = sensitivity_trajectory.description
    assert(isinstance(sensitivity_trajectory_description, SensitivityTerm))
    assert(np.equal(trajectory.timepoints, sensitivity_trajectory.timepoints).all())

    return Trajectory(trajectory.timepoints,
                      trajectory.values + sensitivity_trajectory.values * delta,
                      PerturbedTerm(sensitivity_trajectory_description.ode_term,
                                    sensitivity_trajectory_description.parameter,
                                    delta))
