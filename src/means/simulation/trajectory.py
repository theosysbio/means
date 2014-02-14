import numpy as np

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
        :type timepoints: :class:`numpy.ndarray`
        :param values: values of the curve at each of the timepoints
        :type values: :class:`numpy.ndarray`
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
        return plt.plot(self.timepoints, self.values, *args, label=label, **kwargs)

    def __repr__(self):
        return '{0}({1}, {2}, {3})'.format(self.__class__.__name__, self.timepoints, self.values, self.description)

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
        """
        super(TrajectoryWithSensitivityData, self).__init__(timepoints, values, description)
        self._sensitivity_data = sensitivity_data

    @classmethod
    def from_trajectory(cls, trajectory, sensitivity_data):
        return cls(trajectory.timepoints, trajectory.values, trajectory.description, sensitivity_data)

    @property
    def sensitivity_data(self):
        return self._sensitivity_data