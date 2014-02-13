#!/bin/sh
PYENV_HOME=$WORKSPACE/.pyenv/

CODE_DIR=$WORKSPACE/src
DOCS_DIR=$CODE_DIR/docs

export LD_LIBRARY_PATH=/usr/local/lib # This is needed for assimulo to find sundials lsibraries and thus work properly


# Create virtualenv and install necessary packages
# We might want to add --no-site-packages if this is too slow
virtualenv --python=python2.7 $PYENV_HOME
. $PYENV_HOME/bin/activate
# PYTHON setup insrtructions go below
cd $WORKSPACE

./install_assimulo.sh

# Uninstall previous version of our script
pip uninstall -y means || echo "Means not yet installed"
# Install current version of our script
pip install $CODE_DIR

./run_tests.sh
pylint --rcfile=pylint.rc -f parseable $CODE_DIR/means | tee pylint.out

# Documentation
pip install --quiet sphinx
pip install --quiet sphinx_rtd_theme

cd $DOCS_DIR
make html
