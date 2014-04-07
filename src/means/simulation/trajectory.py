"""
Trajectories
--------

This part of the package provide convenient utilities to manage trajectories.
A :class:`~means.simulation.trajectory.Trajectory` object is generally a time series
containing the values of a given moment (e.g. mean, variance, ...) over a time range.
Trajectories are typically returned by simulations (see :mod:`~means.simulation.simulate` and
:mod:`~means.simulation.ssa`),
or from observation/measurement.

The `TrajectoryCollection` class is a container of trajectories.
It can be used like other containers such as lists.

Both `~means.simulation.trajectory.TrajectoryCollection` and `~means.simulation.trajectory.Trajectory` have there own `.plot()`
method to help representation.
"""

import operator
import numbers
import numpy as np
from means.core.descriptors import Descriptor, Moment
from means.io.serialise import SerialisableObject
from means.simulation import SensitivityTerm
from means.simulation.descriptors import PerturbedTerm


class Trajectory(SerialisableObject):
    """
    A single simulated or observed trajectory for an ODE term.
    """
    _timepoints = None
    _values = None
    _description = None

    yaml_tag = u'!trajectory'

    def __init__(self, timepoints, values, description):
        """

        :param timepoints: timepoints the trajectory was simulated for
        :type timepoints: :class:`iterable`
        :param values: values of the curve at each of the timepoints
        :type values: :class:`iterable`
        :param description: description of the trajectory
        :type description: :class:`~means.core.descriptors.Descriptor`
        """
        self._timepoints = np.array(timepoints)
        self._values = np.array(values)
        self._description = description

        assert(isinstance(description, Descriptor))
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

        :rtype: :class:`~means.core.descriptors.Descriptor`
        """
        return self._description

    def set_description(self, description):
        assert(isinstance(description, Descriptor))
        self._description = description

    def _create_plot(self, *args, **kwargs):
        from matplotlib import pyplot as plt
        # Get label from the kwargs provided, or use self.description as default
        label = kwargs.pop('label', self.description.mathtext())
        # This is needed for matplotlib version 1.1.1
        label = str(label)
        return plt.plot(self.timepoints, self.values, *args, label=label, **kwargs)

    def plot(self, *args, **kwargs):
        """
        Plots the trajectory using :mod:`matplotlib.pyplot`.

        :param args: arguments to pass to :func:`~matplotlib.pyplot.plot`
        :param kwargs: keyword arguments to pass to :func:`~matplotlib.pyplot.plot`
        :return: the result of the :func:`matplotlib.pyplot.plot` function.
        """
        return self._create_plot(*args, **kwargs)

    def _repr_png_(self):
        from IPython.core.pylabtools import print_figure
        from matplotlib import pyplot as plt
        ax = self._create_plot()
        fig = plt.gcf()
        data = print_figure(fig, 'png')
        plt.close(fig)
        return data

    @property
    def png(self):
        from IPython.display import Image
        return Image(self._repr_png_(), embed=True)

    def _repr_svg_(self):
        from IPython.core.pylabtools import print_figure
        from matplotlib import pyplot as plt
        ax = self._create_plot()
        fig = plt.gcf()
        data = print_figure(fig, 'svg')
        plt.close(fig)
        return data

    @property
    def svg(self):
        from IPython.display import SVG
        return SVG(self._repr_png_())

    def resample(self, new_timepoints, extrapolate=False):

        """
        Use linear interpolation to resample trajectory values.
        The new values are interpolated for the provided time points.
        This is generally before comparing or averaging trajectories.

        :param new_timepoints: the new time points
        :param extrapolate: whether extrapolation should be performed when some new time points
            are out of the current time range. if extrapolate=False, it would raise an exception.
        :return: a new trajectory.
        :rtype: :class:`~means.simulation.trajectory.Trajectory`
        """
        if not extrapolate:
            if min(self.timepoints) > min(new_timepoints):
                raise Exception("Some of the new time points are before any time points. If you really want to extrapolate, use `extrapolate=True`")
            if max(self.timepoints) < max(new_timepoints):
                raise Exception("Some of the new time points are after any time points. If you really want to extrapolate, use `extrapolate=True`")
        new_values = np.interp(new_timepoints, self.timepoints, self.values)
        return Trajectory(new_timepoints, new_values, self.description)

    def __repr__(self):

        n_edge_items = 4
        precision = 3

        if len(self.timepoints) <= 2*n_edge_items:
            timepoint_to_print = ", ".join([str(round(i,precision)) for i in self.timepoints])
            values_to_print = ", ".join([str(round(i,precision)) for i in self.values])
        else:
            left_time = ", ".join([str(round(i,precision)) for i in self.timepoints[0: n_edge_items]])
            right_time = ", ".join([str(round(i,precision)) for i in self.timepoints[-n_edge_items: len(self.timepoints)]])

            timepoint_to_print = "{0}, ...,{1}".format(left_time, right_time)
            left_values = ", ".join([str(round(i,precision)) for i in self.values[0: n_edge_items]])
            right_values = ", ".join([str(round(i,precision)) for i in self.values[-n_edge_items: len(self.values)]])
            values_to_print = "{0}, ...,{1}".format(left_values, right_values)

        return '{0} object\ndescription: {1}\ntime points: [{2}]\nvalues: [{3}]'.format(
            self.__class__.__name__,
            self.description,
            timepoint_to_print,
            values_to_print)

    # def __ne__(self, other):
    #     return not self == other


    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return np.equal(self.timepoints, other.timepoints).all() and np.equal(self.values, other.values).all() \
            and self.description == other.description

    def __add__(self, other):
        return self._arithmetic_operation(other, operator.add)
    def __div__(self, other):
        return self._arithmetic_operation(other, operator.div)
    def __mul__(self, other):
        return self._arithmetic_operation(other, operator.mul)
    def __sub__(self, other):
        return self._arithmetic_operation(other, operator.sub)
    def __pow__(self, other):
        return self._arithmetic_operation(other, operator.pow)

    def __radd__(self, other):
        # for `sum()`    to work
        return self + other


    def _arithmetic_operation(self, other, operation):
        """
        Applies an operation between the values of a trajectories and a scalar or between
        the respective values of two trajectories. In the latter case, trajectories should have
        equal descriptions and time points
        """
        if isinstance(other, Trajectory):
            if self.description != other.description:
                raise Exception("Cannot add trajectories with different descriptions")
            if not np.array_equal(self.timepoints, other.timepoints):
                raise Exception("Cannot add trajectories with different time points")
            new_values = operation(self.values, other.values)
        elif isinstance(other, numbers.Real):
            new_values = operation(self.values, float(other))
        else:
            raise Exception("Arithmetic operations is between two `Trajectory` objects or a `Trajectory` and a scalar.")

        return Trajectory(self.timepoints, new_values, self.description)


    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = [('timepoints', data.timepoints),
                   ('values', data.values),
                   ('description', data.description)]

        return dumper.represent_mapping(cls.yaml_tag, mapping)

class TrajectoryWithSensitivityData(Trajectory):
    """
    An extension to :class:`~means.simulation.Trajectory` that provides data about the sensitivity
    of said trajectory as well.

    """

    _sensitivity_data = None
    yaml_tag = '!trajectory-with-sensitivity'

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
        self._sensitivity_data = TrajectoryCollection(sensitivity_data)

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
        return isinstance(other, self.__class__) and \
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
            # This is needed for matplotlib version 1.1.1
            label = str(label)
            # Fill_between does not generate a legend entry, use this hack with Rectangle to do this
            plt.gca().add_patch(plt.Rectangle((0, 0), 0, 0, alpha=alpha,
                                              label=label,
                                                    *args, **kwargs))

    def _arithmetic_operation(self, other, operation):
        """
        Applies an operation between the values of a trajectories and a scalar or between
        the respective values of two trajectories. In the latter case, trajectories should have
        equal descriptions and time points
        """
        if isinstance(other, TrajectoryWithSensitivityData):
            if self.description != other.description:
                raise Exception("Cannot add trajectories with different descriptions")
            if not np.array_equal(self.timepoints, other.timepoints):
                raise Exception("Cannot add trajectories with different time points")
            new_values = operation(self.values, other.values)
            new_sensitivity_data = [operation(ssd, osd) for ssd, osd in
                                    zip(self.sensitivity_data, other.sensitivity_data)]

        elif isinstance(other, numbers.Real):
            new_values = operation(self.values, float(other))
            new_sensitivity_data = [operation(ssd, float(other)) for ssd in self.sensitivity_data]

        else:
            raise Exception("Arithmetic operations is between two `TrajectoryWithSensitivityData`\
                            objects or a `TrajectoryWithSensitivityData` and a scalar.")

        return TrajectoryWithSensitivityData(self.timepoints, new_values, self.description, new_sensitivity_data )


    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = [('timepoints', data.timepoints),
                   ('values', data.values),
                   ('description', data.description),
                   ('sensitivity_data', data.sensitivity_data)]
        return dumper.represent_mapping(cls.yaml_tag, mapping)


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


class TrajectoryCollection(SerialisableObject):
    """
    A container of trajectories with representation functions for matplotlib and IPythonNoteBook.
    In most cases, it simply behaves as list.
    """

    yaml_tag = '!trajectory-collection'

    trajectories = None

    def __init__(self, trajectories):
        # Hack to allow passing instantiated TrajectoryCollection objects as well
        if isinstance(trajectories, self.__class__):
            trajectories = trajectories.trajectories
        self._trajectories = trajectories

    @property
    def trajectories(self):
        """
        Return a list of all trajectories in the collection
        :rtype: list[:class:`~means.simulation.trajectory.Trajectory`]
        """
        return self._trajectories

    def __iter__(self):
        return iter(self.trajectories)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, item):
        answer = self.trajectories[item]
        if isinstance(answer, list):
            # Wrap around self class if we return a list of trajectories
            return self.__class__(answer)
        else:
            return answer

    def _create_figure(self):

        def _key_and_title(description):
            if isinstance(description, Moment):
                key = (description.__class__, description.order)
                title = 'Moments of order {0}'.format(description.order)
            else:
                key = description.__class__
                title = description.__class__.__name__
            return key, title

        from matplotlib import pyplot as plt


        subplot_numbers = {}
        subplot_counter = 0
        for trajectory in self.trajectories:
            description = trajectory.description
            key, title = _key_and_title(description)

            try:
                subplot_number = subplot_numbers[key]
            except KeyError:
                subplot_counter += 1
                subplot_number = subplot_counter
                subplot_numbers[key] = subplot_number

        total_subplots = subplot_counter

        for i,trajectory in enumerate(self.trajectories):
            description = trajectory.description
            key, title = _key_and_title(description)

            subplot_number = subplot_numbers[key]

            plt.subplot(total_subplots, 1, subplot_number)
            plt.title(title)
            trajectory.plot()
            plt.legend(bbox_to_anchor=(1, 1), loc=2, ncol=2)

            if i == len(self.trajectories) - 1:
                plt.xlabel('time')

        return plt.gcf()

    def plot(self):
        self._create_figure()

    def _repr_png_(self):
        from IPython.core.pylabtools import print_figure
        from matplotlib import pyplot as plt
        fig = self._create_figure()
        data = print_figure(fig, 'png')
        plt.close(fig)
        return data

    @property
    def png(self):
        from IPython.display import Image
        return Image(self._repr_png_(), embed=True)

    def _repr_svg_(self):
        from IPython.core.pylabtools import print_figure
        from matplotlib import pyplot as plt
        fig = self._create_figure()
        data = print_figure(fig, 'svg')
        plt.close(fig)
        return data

    @property
    def svg(self):
        from IPython.display import SVG
        return SVG(self._repr_png_())

    def __unicode__(self):
        return u"<{self.__class__.__name__}>\n{self.trajectories!r}".format(self=self)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __repr__(self):
        return str(self)

    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = {'trajectories': data.trajectories}
        return dumper.represent_mapping(cls.yaml_tag, mapping)


    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.trajectories == other.trajectories

    def __ne__(self, other):
        return not self == other

