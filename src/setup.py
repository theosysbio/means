from distutils.core import setup

setup(
    name='means',
    version='trunk',
    author='Sisi Fan, Quentin Geissmann, Saulius Lukauskas',
    author_email= 'sisi.fan10@imperial.ac.uk, quentin.geissmann13@imperial.ac.uk, saulius.lukauskas13@imperial.ac.uk',
    packages=['means',
              'means.approximation',
              'means.approximation.lna',
              'means.approximation.mea',
              'means.core',
              'means.examples',
              'means.inference',
              'means.io',
              'means.simulation',
              'means.tests',
              'means.util',
              'means.pipes'],
    scripts=['bin/means'],
    url=None, # TODO: url
    license=None, # TODO: license
    description='Moment Expansion Approximation method implementation with simulation and inference packages',
    long_description=open('README.txt').read(),
    extras_require={
        'pipes': ['luigi>=1.0.13'],
    },
    install_requires=[
        "numpy>=1.6.1",
        "sympy>=0.7.4.1",
        "matplotlib>=1.1.0",
        "scipy>=0.10.1",
        "PyYAML>=3.10",
        "Assimulo>=2.5.1",
    ],
)
