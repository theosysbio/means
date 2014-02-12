#!/bin/bash
ASSIMULO_TRUNK=$WORKSPACE/.assimulo-trunk/
if [ -d $ASSIMULO_TRUNK ]; then
     svn update $ASSIMULO_TRUNK || echo "Warning: Could not update assimulo"
else
    svn checkout --trust-server-cert https://svn.jmodelica.org/assimulo/trunk $ASSIMULO_TRUNK
fi

# assimulo setup
export LD_LIBRARY_PATH=/usr/local/lib # This is needed for assimulo to find sundials lsibraries and thus work properly
pip install --quiet cython # needed for assimulo
pip install --quiet numpy # also needed for assimulo
cd $ASSIMULO_TRUNK
python setup.py install
