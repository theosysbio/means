#!/bin/sh
PYENV_HOME=$WORKSPACE/.pyenv/
ASSIMULO_TRUNK=$WORKSPACE/.assimulo-trunk/
CODE_DIR=$WORKSPACE/src
DOCS_DIR=$CODE_DIR/docs

# Delete previously built virtualenv
#if [ -d $PYENV_HOME ]; then
#    rm -rf $PYENV_HOME
#fi
if [ -d $ASSIMULO_TRUNK ]; then
     svn update $ASSIMULO_TRUNK || echo "Warning: Could not update assimulo"
else 
    svn checkout --trust-server-cert https://svn.jmodelica.org/assimulo/trunk $ASSIMULO_TRUNK
fi
ls $WORKSPACE
# Create virtualenv and install necessary packages
# We might want to add --no-site-packages if this is too slow
virtualenv --python=python2.7 $PYENV_HOME
. $PYENV_HOME/bin/activate
# PYTHON setup insrtructions go below

# assimulo setup
export LD_LIBRARY_PATH=/usr/local/lib # This is needed for assimulo to find sundials lsibraries and thus work properly
pip install --quiet cython # needed for assimulo
cd $ASSIMULO_TRUNK
python setup.py install

cd $WORKSPACE

pip install --quiet sphinx
pip install --quiet sphinx_rtd_theme

# Uninstall previous version of our script
pip uninstall -y means || echo "Means not yet installed"
# Install current version of our script
pip install $CODE_DIR

./run_tests.sh
pylint -f parseable $CODE_DIR/means | tee pylint.out

cd $DOCS_DIR
make html
