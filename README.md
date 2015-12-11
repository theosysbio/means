MEANS
==========
Moment Expansion Approximation method implementation with simulation and inference packages.

Dependencies
==============

Sundials
--------------
Prior to installation of this package ensure [`sundials`](https://computation.llnl.gov/casc/sundials/main.html)
libraries are installed on your system.

The exact steps to install them will depend on your system and configuration.

Generally, the easiest way to install these packages is via the system's package manager.
For instance, on Mac OS X these libraries can be installed via [HomeBrew](http://brew.sh/):

    brew install sundials

If you are using ubuntu, you can use `apt-get` to install the packages:

    apt-get install libsundials-cvode1 libsundials-cvodes2 libsundials-ida2 libsundials-idas0 libsundials-kinsol1 libsundials-nvecserial0 libsundials-serial libsundials-serial-dev

Users of `anaconda` python distribution, could use the [recipe provided by `chemreac`](https://anaconda.org/chemreac/sundials) to install it:

    conda install -c https://conda.anaconda.org/chemreac sundials

At this point, make sure to note the path where the package manager installs the libraries to.
For instance, `apt-get` is likely to put the libraries to `/usr/lib` whereas anaconda is likely to put it in `/home/username/anaconda/lib`. Ensure this lib directory is in `LD_LIBRARY_PATH` variable.

    export LD_LIBRARY_PATH="/your/sundials/home/lib:$LD_LIBRARY_PATH"

Also note the directory prefix without the `lib` as you might need to specify this path as `--sundials-home`` when istalling `assimulo` (see the [troubleshooting section below](#symptom-tests-fail-with-no-module-named-sundials)).

On other linux systems one might need to compile the sundials libraries from source.
Visit [`sundials` homepage](https://computation.llnl.gov/casc/sundials/download/download.php) for 
instructions on how to obtain the latest release of the library.

For the newest release of `sundials`, follow instructions in the pdf distributed with the `sundials` source.
Ensure that `-DCMAKE_C_FLAGS=-fPIC` is set when installing.

If you are installing an older version of the library, then make sure that

`--enable-shared`, `--enable-static`, `--with-gnu-ld` 
and `--with-cflags=-fPIC` flags are set.
 
Similarly, ensure that the `--prefix` is set to `/usr/local`, or alternatively see [troubleshooting section below](#symptom-tests-fail-with-no-module-named-sundials).

Installation (Development version)
============

Once sundials libraries are installed correctly.

The development version of this package can be installed by downloading the code, navigating to `src`
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

No tests should fail if installation was successful. And one can proceed to the tutorial.

Troubleshooting
===================

Symptom: Tests fail with `No module named sundials`
---------------------------------------------
This error indicates that `assimulo` installation failed to find the sundials in the system path.
Please ensure that you set appropriate flags in `sundials` installation, as described in previous section and attempt
to reinstall assimulo as follows:

```
pip uninstall assimulo
pip install assimulo
```

If you installed the `sundials` libraries to prefix other than `/usr/local` (for instance, 
ArchLinux installs the library to `/usr` by default), install assimulo as follows:

```
pip install assimulo --install-option="--sundials-home=/your/sundials/installation/prefix"
```
Replacing the `/your/sundials/installation/prefix` with the appropriate prefix.

Once this is done try running tests again.

Symptom: Tests fail with `ImportError: /usr/lib/libsundials_idas.so.0: undefined symbol: dscal_`
------------------------------------------------------------------------------------------------
The symptom above indicates incorrect linkage between `libsundials_idas.so` and BLAS/LAPACK libraries.
A workaround for this issue is to add `libblas.so liblapack.so` to `LD_PRELOAD` environment variable.
To do this, make sure to set this environment variable first:

```
export LD_PRELOAD='libblas.so liblapack.so'
```

And then run everything else `python setup.py test` as usual.
