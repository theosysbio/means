#!/bin/bash
EXAMPLES_DIR=$WORKSPACE/prototypes/examples

pip install --quiet nosexcover
pip install --quiet pylint

if [ "$1" == "with-slow-tests" ]; then
   SLOWTESTS="--no-skip"
else
   SLOWTESTS=""
fi
nosetests $SLOWTESTS --with-xcoverage --with-xunit --cover-package=means --cover-erase $CODE_DIR
nosetests --with-xunit $EXAMPLES_DIR
cd $INOUT_DIR

