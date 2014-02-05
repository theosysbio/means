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
              'means.inference',
              'means.model',
              'means.simulation',
              'means.util',
              'means.tests'],
    package_src={'means' : 'src/means'},
    scripts=['bin/means'],
    url=None, # TODO: url
    license=None, # TODO: license
    description='Moment Expansion Approximation method implementation with simulation and inference packages',
    long_description=None, # TODO: readme.txt
    install_requires=[
        "numpy>=1.8.0",
        "sympy>=0.7.2",
        "matplotlib>=1.1.1",
        "scipy>=0.13.2",
        "Assimulo==trunk"
    ],
)