"""
MEANS package comes with built-in support for data processing pipelines.
The pipeline support is built on top of `luigi package by Spotify`_,
which provides a minimal dependency specification framework as well as centralised scheduler.

It is strongly encouraged to familiarise yourself with :mod:`luigi`'s tutorial before attempting to use the
:mod:`means.pipes` package.
This tutorial is available from github at https://github.com/spotify/luigi/blob/master/README.md.

The main :mod:`luigi` workflow consists of defining classes inheriting from :mod:`luigi.task` and defining
:meth:`requires()`, :meth:`output()` and :meth:`run()` methods that describe the task dependencies, output and
how to run the task respectively.

Automatic Serialisation
-----------------------------

:class:`means.pipes.Task` simplifies this behaviour by defining :meth:`output()` method for you. The said output method
picks a filename for the output automatically. The default filepath template is ``<OUTPUT_DIR>/<CLASS>/<FILENAME>``
where ``<OUTPUT_DIR>`` is set to be ``task-output`` by default, this can be changed by specifying `output > directory`
directive in the `configuration file`_, e.g.: ::

    [output]
    directory=some_other_directory

The ``<CLASS>`` placeholder is set to the name of the class (so it is easier to copy/delete all the output files
for some task), and the ``<FILENAME>`` is automatically set to the name of the class followed by dash-separated list
of parameters, i.e. ``MEATask-p53-1-scalar-True.pickle``.

The output is serialised using :mod:`pickle` using
`pickle.HIGHEST_PROTOCOL` by default. This means that, unlike the files serialised using :func:`~means.io.to_file`,
the output files are not human-readable, but more efficient to machine read and write.

This serialisation is done with the help of :class:`~means.pipes.targets.PickleSerialiser` and
:class:`~means.pipes.targets.PickleSerialiserWithAdditionalParameters` targets.
The former just stores the raw data in a file, while the latter allows additional parameters, such as the runtime of the
task to be stored along with the data. The latter is used in all of the tasks by default, and the aforementioned runtime
is stored in the file along with the data.

To read the data from the serialised files, create a new instance
of the :class:`~means.pipes.targets.PickleSerialiserWithAdditionalParameters` object,
and call :meth:`~means.pipes.targets.PickleSerialiserWithAdditionalParameters.load()` method.
You can access the runtime of the task by providing ``'runtime'`` as the first parameter.

Note that there is no need to create a new serialiser object in the `run()` method of the task.
These instances are already provided by :meth:`input()` method defined in the class.
This method returns the instances of output methods of the classes and allows you to work with them.
Therefore you would just need to call the `load()` method only. In fact, both pickle serialisers cache their output
in memory, so the files do not need to be re-read every-time a new task might need to do so, using the ``input()``
method would ensure all instances of the object point to the same memory location, and thus to the same cache,
something that cannot be guaranteed when new instances of the object are created.

:meth:`~means.pipes.tasks.Task._return_object()` is the new ``run()``
---------------------------------------------------------------------
Since most of what tasks do is to run some code and store the output in the file,
:mod:`means.pipes` implement that part of the code for you.
:meth:`~means.pipes.tasks.Task.run()` calls the function :meth:`~means.pipes.tasks.Task._return_object()` that you
should define in all of the tasks and stores the result in aforementioned output file. The result is stored together
with the runtime.

This means that the main logic of the task should go into this function, and the object that is returned should be
the output you want to store. Note that you can always return a tuple, of dictionary,
if you have multiple things to output from a task, for instance here's an example how a minimal task could look ::

    >>> from means.pipes import Task
    >>> class MyTask(Task):
    ...     def _return_object(self):
    ...         a = 'foo'
    ...         b = 'bar'
    ...
    ...         # This would store `a` and `b` as a tuple in the output file `<output_dir>/MyTask/MyTask.pickle'
    ...         return a, b

Generating figures
------------------
:mod:`means.pipes` provides a :class:`~means.pipes.tasks.FigureTask()` object that automatically serialises
:mod:`matplotlib` figures in `svg` or `pdf` formats (specified by ``figure_format`` parameter).
To use these, define your tasks to inherit from :class:`~means.pipes.tasks.FigureTask` and return a
:class:`matplotlib.Figure` object from the :meth:`~means.pipes.tasks.FigureTask._return_object()`, e.g. ::

    >>> from means.pipes import FigureTask
    >>> class MyFigure(FigureTask):
    ...
    ...     def _return_object(self):
    ...         from matplotlib import pyplot as plt
    ...         # Create a figure
    ...         fig = plt.figure()
    ...
    ...         # Plot something
    ...         ax = plt.subplot(1,1,1)
    ...         ax.plot([1,2], [3,4], label='foobar')
    ...         fig.legend()
    ...
    ...         # just return the figure, do not issue ``plt.show()`` or anything of that sort
    ...         return fig

The example above would save the figure to ``<output-directory>/MyFigure/MyFigure-pdf.pdf``
Note that there is a known issue with :class:`~means.pipes.tasks.FigureTask` and luigi runtime with multiple workers.
The tasks will likely fail in multiple workers, however, this can be rectified by rerunning them using single worker
only. This is not a big problem as the tasks that the figures depend on (which are likely time-consuming)
will be completed already before the :class:`~means.pipes.tasks.FigureTask` have a chance to fail, therefore only
these tasks will need to be run.

Please also note that there is no way to load the figure results from the tasks as they have been rasterised into
different formats by the time they have been saved, if you need this, it is advised to use
:class:`~means.pipes.tasks.Task` rather than :class:`~means.pipes.tasks.FigureTask` as the former uses pickle
serialisation, which should be reversable.

Joining figures into LaTeX subfigures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:mod:`means.pipes` defines a task that can join all it's dependancies into a LaTeX subfigure and output the resulting
`tex` file. This task is called :class:`~means.pipes.tasks.TexFigureTask`. And it will put the results of all of the
figures in it's ``requires()`` method, into a latex figure. The :attr:`~means.pipes.tasks.TexFigureTask.standalone``
parameter, if set to true, will make a standalone (directly compilable to latex) figure, whereas, if it is set to
false, it will create a figure that would need to be imported to the document some other way before compilation.

Note that some of the formats :class:`~means.pipes.tasks.FigureTask` supports might not be supported in LaTeX, and
vice versa.

Predefined tasks for :mod:`means`-specific needs
------------------------------------------------
The pipeline support for MEANS comes with batteries included and predefines tasks for the common means operations
such as simulation and generation of problems using the approximation methods.

Please see :mod:`means.pipes.tasks` documentation for the list of such classes.

Other considerations
-----------------------

Caching
~~~~~~~
MEANS package uses clever caching of the C expression evaluators generated using `sympy`.
Due to the way `luigi` workers create a sandbox for each tasks execution, this caching is not persistent between
the tasks run by the default :class:`luigi.Worker`. To overcome this, means define their own
:class:`~means.pipes.interface.PreloadingWorker` that looks for tasks with
:class:`~means.pipes.interface.TaskPreloadingHint` and calls the
:meth:`~means.pipes.interface.TaskPreloadingHint.preload()` method they must define before the establishment of
the sandbox. This allows the :class:`~means.pipes.tasks.TrajectoryTask` to ensure that the efficient evaluators
are cached for all tasks and compiled only once, speeding up the code vastly.

In order for this optimisation, to work, one must always import the run method from :mod:`means.pipes.interface` package
and not from `luigi`, i.e. ::

    >>> from means.pipes import run
    >>> if __name__ == '__main__':
    ...    run()

Objects with long string representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Some objects, such as :class:`means.Model` have long `str()` representations designed to be helpful in interactive mode.
Since :mod:`means.pipes` uses the `str()` method to generate the filenames, these long methods cause said filenames
become incredibly long. When filenames become longer than 200 characters,
 :mod:`means.pipes` shortens the filenames by hashing the parameters using `md5` algorithm,
 what makes them unreadable by humans.

In order to preserve the readability of filenames, you can create an object that inherits from the original one, and
changes it's ``__str__`` function to some other readable representation, for instance,
we could create `P53Model()` class that would initialise the `P53` model from :mod:`means.examples` and assign
a friendly short-name for it as follows::

    >>> class P53Model(means.Model):
    ...     def __init__(self):
    ...         from means.examples import MODEL_P53
    ...         super(P53Model, self).__init__(MODEL_P53.species, MODEL_P53.parameters, MODEL_P53.propensities,
    ...                                        MODEL_P53.stoichiometry_matrix)
    ...
    ...     def __str__(self):
    ...         # Override the str() methods so they do not print the whole blerch of things, but
    ...         # only a nice and easily readable "p53"
    ...         return 'p53'

.. _`luigi package by Spotify`: https://github.com/spotify/luigi
.. _`configuration file`: https://github.com/spotify/luigi/blob/master/README.md#configuration

"""

try:
    import luigi
except ImportError:
    raise ImportError('means.pipes requires luigi package. Please install it before using means.pipes')

from interface import TaskPreloadingHint, PreloadingWorker, PreloadingWorkerSchedulerFactory, run
from parameters import *
from targets import PickleSerialiserWithAdditionalParameters, PickleSerialiser
from tasks import Task, MEATask, TrajectoryTask, TaskBase, FigureTask, TexFigureTask, SSATrajectoryTask,InferenceTask

import interface, parameters, targets, tasks