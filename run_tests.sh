#!/bin/bash
EXAMPLES_DIR=$WORKSPACE/prototypes/examples

pip install --quiet nosexcover
pip install --quiet pylint
# Needed to run nosetests for notebooks
pip install --quiet IPython
pip install --quiet pyzmq

if [ "$1" == "with-slow-tests" ]; then
   SLOWTESTS="--no-skip"
else
   SLOWTESTS=""
fi
nosetests $SLOWTESTS --with-xcoverage --with-xunit --cover-package=means --cover-erase $CODE_DIR
nosetests --with-xunit --xunit-file="nosetests-notebooks.xml" $EXAMPLES_DIR
cd $INOUT_DIR

