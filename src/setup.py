from setuptools import setup, find_packages
import os

DESCRIPTION = """
MEANS: python package for Moment Expansion Approximation, iNference and Simulation

A free, user-friendly tool implementing an efficient moment expansion approximation with parametric closures
that integrates well with the IPython interactive environment.
Our package enables the analysis of complex stochastic systems without any constraints
on the number of species and moments studied and the type of rate laws in the system.
In addition to the approximation method our package provides numerous tools to help
non-expert users in stochastic analysis.
"""

# -- Assimulo dependency checks ---------------------------------------------------------------------
# numpy and cython are required for assimulo installation, yet their setup.py is badly done
# so the setup fails. Add these checks so our error message is displayed instead
# can be removed once assimulo ups their game.
try:
    import cython
except ImportError:
    raise ImportError('Please install cython first `pip install cython`')

try:
    import numpy
except ImportError:
    raise ImportError('Please install numpy first `pip install numpy`')
# ---------------------------------------------------------------------------------------------------------
setup(
    name='means',
    version='1.0.0',

    description='Moment Expansion Approximation method implementation with simulation and inference packages',
    long_description=DESCRIPTION,

    author='Sisi Fan, Quentin Geissmann, Eszter Lakatos, Saulius Lukauskas, '
           'Angelique Ale, Ann C. Babtie, Paul D.W. Kirk, Michael P.H. Stumpf',
    author_email='m.stumpf@imperial.ac.uk,e.lakatos13@imperial.ac.uk',

    url='https://github.com/theosysbio/means',
    license='MIT',

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering'
    ],
    keywords=['moment expansion', 'approximation', 'simulation', 'inference'],

    packages=find_packages(),

    setup_requires=[
        'numpy>=1.6.1',  # Numpy has to be installed before others
    ],
    install_requires=[
        "numpy>=1.6.1",
        "sympy>=0.7.5",
        "matplotlib>=1.1.0",
        "scipy>=0.10.1",
        "PyYAML>=3.10",
        "Assimulo>=2.5.1",
    ],
    tests_require=['nose'],
    test_suite='nose.collector'
)
