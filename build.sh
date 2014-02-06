PYENV_HOME=$WORKSPACE/.pyenv/
ASSIMULO_TRUNK=$WORKSPACE/.assimulo-trunk/
CODE_DIR=$WORKSPACE/MEA_package/ProgramFiles
REQUIREMENTS_TXT=$WORKSPACE/MEA_package/requirements.txt

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

pip install --quiet nosexcover
pip install --quiet pylint

# Uninstall previous version of our script
pip uninstall -y means || echo "Means not yet installed"
# Install current version of our script
pip install $CODE_DIR

nosetests --with-xcoverage --with-xunit --cover-package=means --cover-erase $CODE_DIR 
pylint -f parseable $CODE_DIR/src/means | tee pylint.out
cd $CODE_DIR
python -m means.tests.regression_tests --xunit | tee $WORKSPACE/regression_tests.xml
