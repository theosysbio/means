#!/bin/bash
EXAMPLES_DIR=$WORKSPACE/prototypes/examples

pip install --quiet nosexcover
pip install --quiet pylint
# Needed to run nosetests for notebooks
pip install --quiet IPython
pip install --quiet pyzmq
pip install --quiet pypng
pip install --quiet ipycache

if [ "$1" == "with-slow-tests" ]; then
   SLOWTESTS="--no-skip"
else
   SLOWTESTS=""
fi
echo "Running tests"
nosetests $SLOWTESTS --with-xcoverage --with-xunit --cover-package=means --cover-erase $CODE_DIR
echo "Running notebook tests"
# Run notebook tests
nosetests --with-xunit --xunit-file="notebook-tests.xml" $EXAMPLES_DIR

