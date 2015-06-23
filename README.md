MEANS
=========
Moment Expansion Approximation method implementation with simulation and inference packages.

Dependencies
==============

Sundials
--------------
Prior to installation of this package ensure [`sundials`](https://computation.llnl.gov/casc/sundials/main.html)
libraries are installed on your system.

On Mac OS X these libraries can be installed via [HomeBrew](http://brew.sh/):

    brew install sundials

On other linux systems one might need to compile the sundials libraries from source.
Visit [`sundials` homepage](https://computation.llnl.gov/casc/sundials/download/download.php) f
or instructions on how to obtain the latest release of the library.

When installing the packages, ensure the following flags are set:
```
    ./configure --disable-debug --disable-dependency-tracking --prefix=/usr/local --enable-shared --enable-static --with-gnu-ld --with-cflags=-fPIC
```
Then run (as root):
```
    make && make install
```

Once the libraries are installed, make sure they are in your `LD_LIBRARY_PATH`, i.e. add the following to your `.bashrc`
```
    export LD_LIBRARY_PATH=/usr/local/lib:LD_LIBRARY_PATH
```
as otherwise the software will fail to link properly.

Installation (Development version)
============

The development version of this package can be installed by checking out the code, navigating to `src`
directory and typing::
```
    pip install -e .
```

Due to the way ``pip`` handles dependencies, you might need to install ``cython``, ``numpy`` prior to `means`:

```
    pip install cython
    pip install numpy
```

Verifying Installation and Running Tests
==============

Since a lot of things can go wrong during the installation, it is important to verify everything by running the test suite.
This can be done by running the test command in the setup suite:

```
    python setup.py test
```

No tests should fail if installation was successful.

If you see an error message similar to:

```
ImportError: No module named sundials
```

it is likely that the installation of `sundials` libraries failed.

Please repeat the `sundials` installation steps described above, and reinstall `assimulo` package afterwards:

```
pip uninstall assimulo
pip install assimulo
```

Once this is done try running tests again.