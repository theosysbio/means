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
missing_dependicies = []

try:
    import cython
except ImportError:
    missing_dependicies.append('Cython')

try:
    import numpy
except ImportError:
    missing_dependicies.append('numpy')

if missing_dependicies:
    raise ImportError('Please install {} first: `pip install {}`'.format(
        ' and '.join(missing_dependicies),
        ' '.join(missing_dependicies)
    ))
# ---------------------------------------------------------------------------------------------------------
setup(
    name='means',
    version='1.0.1-dev',

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
        "Assimulo>=2.8",
    ],
    tests_require=['nose'],
    test_suite='nose.collector'
)
