from setuptools import setup, find_packages


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
    version='0.0.1',

    description='Moment Expansion Approximation method implementation with simulation and inference packages',
    long_description=open('README.txt').read(),

    author='Sisi Fan, Quentin Geissmann, Saulius Lukauskas',
    author_email='sisi.fan10@imperial.ac.uk, quentin.geissmann13@imperial.ac.uk, saulius.lukauskas13@imperial.ac.uk',

    url=None,  # TODO: url
    license=None,  # TODO: license

    classifiers=[],  # TODO: classifiers
    keywords='',  # TODO: keywords

    packages=find_packages(),

    extras_require={
        'pipes': ['luigi>=1.0.13'],
    },
    setup_requires=[
        'numpy>=1.6.1',  # Numpy has to be installed before others
    ],
    install_requires=[
        "numpy>=1.6.1",
        "sympy>=0.7.4.1",
        "matplotlib>=1.1.0",
        "scipy>=0.10.1",
        "PyYAML>=3.10",
        "Assimulo>=2.5.1",
    ],
)
