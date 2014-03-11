#!/bin/bash
ASSIMULO_TRUNK=$WORKSPACE/.assimulo-trunk/
echo "Installing assimulo: "
echo `which python`

if [ -f "assimulo-rev" ]; then
   ASSIMULO_REV=`cat assimulo-rev`
fi
if [ -d $ASSIMULO_TRUNK ]; then
     svn update $ASSIMULO_TRUNK $ASSIMULO_REV || echo "Warning: Could not update assimulo"
else
    svn checkout --trust-server-cert https://svn.jmodelica.org/assimulo/trunk $ASSIMULO_TRUNK
fi

# assimulo setup
pip install --quiet cython # needed for assimulo
pip install --quiet numpy # also needed for assimulo
cd $ASSIMULO_TRUNK
python setup.py install
