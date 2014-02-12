from distutils.core import setup

setup(
    name='means',
    version='trunk',
    author='Theoretical Systems Biology Group',
    author_email=None, # TODO: author email
    packages=['means',
              'means.approximation',
              'means.approximation.lna',
              'means.approximation.mea',
              'means.examples',
              'means.inference',
              'means.model',
              'means.plotting',
              'means.simulation',
              'means.util',
              'means.tests'],
    scripts=['bin/means'],
    url=None, # TODO: url
    license=None, # TODO: license
    description='Moment Expansion Approximation method implementation with simulation and inference packages',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy>=1.6.1",
        "sympy>=0.7.1",
        "matplotlib>=1.1.0",
        "scipy>=0.10.0",
        "Assimulo==trunk"
    ],
)