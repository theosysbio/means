==========
Means
==========

Dependancies
==============

Sundials
--------------
This package relies on the installation of `sundials`_ libraries on your system.

On Mac OS X these libraries can be installed via HomeBrew_::

    brew install sundials

On other linux systems one might need to compile the sundials libraries from source.
Make sure the following flags are set when compiling these libraries::

    ./configure --disable-debug --disable-dependency-tracking --prefix=/usr/local --enable-shared --enable-static --with-gnu-ld --with-cflags=-fPIC

Once the libraries are installed, make sure they are in your `LD_LIBRARY_PATH`, i.e.::

    export LD_LIBRARY_PATH=/usr/local/lib:LD_LIBRARY_PATH

.. HomeBrew_: http://brew.sh/
.. `sundials`_: https://computation.llnl.gov/casc/sundials/main.html

Assimulo
-----------
This package relies on development version of `assimulo` package.
This version must be installed from source as `pip` will not be able to handle this.
To install the development version of this package, make sure your sundials library installation succeeded, then
checkout the development version of the code::

    svn checkout --trust-server-cert https://svn.jmodelica.org/assimulo/trunk assimulo-trunk

Navigate to recently checked out directory `assimulo-trunk` and install the package::

    cd assimulo-trink
    python setup.py install --sundials-home=/usr/local

Verify whether the installation succeeded by trying out one of the examples in the package, i.e.::

    $ python -i
    >>> from assimulo.examples.cvode_basic import run_example
    >>> run_example()

Installation (Development version)
============

The development version of this package can be installed by checking out the code, navigating to `ProgramFiles`
directory and typing::

    pip install -e .

Running tests
==============
All tests for the package are available in `means.tests` package that is installed together with the rest.
In order to run regression tests type::

    python -m means.tests.regression_tests

From `ProgramFiles` directory (or any other directory with access to `InOutput` dir)