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
Visit [`sundials` homepage](https://computation.llnl.gov/casc/sundials/download/download.php) for 
instructions on how to obtain the latest release of the library.

Follow instructions in the pdf distributed with the `sundials` source.
Ensure that `-DCMAKE_C_FLAGS=-fPIC` is set when installing.
Pay a particular attention to the installation prefix `-DCMAKE_INSTALL_PREFIX` parameter, which needs
 to be set to `/usr/local` for installation of `assimulo` package, or, alternatively, follow the troubleshooting steps
 at the end of this README file.
 
TODO: these are incomplete, but I give up for now.

If you are installing an older version of the library, ensure that

`--enable-shared`, `--enable-static`, `--with-gnu-ld` 
and `--with-cflags=-fPIC` flags are set.
 
Similarly, ensure that the `--prefix` is set to `/usr/local`, or alternatively see troubleshooting section below.
An example `configure` command that sets the appropriate parameters:
```
    ./configure --prefix=/usr/local --enable-shared --enable-static --with-gnu-ld --with-cflags=-fPIC
```
Then run (as root):
```
    make && make install
```

Once the libraries are installed, make sure they are in your `LD_LIBRARY_PATH` as otherwise
 the software will fail to link properly. For instance, add the following to your `.bashrc`
```
    export LD_LIBRARY_PATH=/usr/local/lib:LD_LIBRARY_PATH
```
Note how `/usr/local/lib` matches `--prefix` parameter above. 

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

Troubleshooting
===================

Symptom: Assimulo installation fails when using latest `sundials` library
---------------------------------
Particularly, `error: ‘struct KINMemRec’ has no member named ‘kin_sfdotJp’` might be hiding somewhere in the output.

TODO: not sure how to solve this yet, give up for now.

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
TODO: this is probably caused by incorrect linking with lapack library. Cannot debug it now as I need to do something else.
