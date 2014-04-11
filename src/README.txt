==========
Means
==========

Dependencies
==============

Sundials
--------------
This package relies on the installation of `sundials`_ libraries on your system.

On Mac OS X these libraries can be installed via HomeBrew_::

    brew install sundials

On other linux systems one might need to compile the sundials libraries from source.
You can download and untar the source using, for instance:

    wget http://computation.llnl.gov/casc/sundials/download/code/sundials-2.5.0.tar.gz
    tar -xvf sundials-2.5.0.tar.gz && cd sundials-2.5.0

Make sure the following flags are set when compiling these libraries::

    ./configure --disable-debug --disable-dependency-tracking --prefix=/usr/local --enable-shared --enable-static --with-gnu-ld --with-cflags=-fPIC

Then run (as root):
    make && make install

Once the libraries are installed, make sure they are in your `LD_LIBRARY_PATH`, i.e.::

    export LD_LIBRARY_PATH=/usr/local/lib:LD_LIBRARY_PATH

.. HomeBrew_: http://brew.sh/
.. `sundials`_: https://computation.llnl.gov/casc/sundials/main.html


Installation (Development version)
============

The development version of this package can be installed by checking out the code, navigating to `src`
directory and typing::

    pip install -e .

Due to the way ``pip`` handles dependencies, you might need to install ``cython``, ``numpy``, ``scipy`` and ``matplotlib``
separately before installing ``means``::

    pip install cython
    pip install numpy
    pip install scipy
    pip install matplotlib

Running tests
==============
All tests for the package are available in `means.tests` package that is installed together with the rest.
In order to run these tests, one could just use `nose` package and type the following to the command line::

    nosetests src/tests

