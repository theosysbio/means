#!/bin/bash

##  The expected out files are named`MODEL_FILENAME.txt.out`
INOUT_DIR="../Inoutput/"
OUT_FILE="ODEout.tmp"

testModel(){
    parameters=$1
    expected_output=$2

    echo "> python runprogram $parameters --ODEout=$OUT_FILE"	
	time python runprogram.py $parameters --ODEout=$OUT_FILE

    if [ ! -f $INOUT_DIR/$OUT_FILE ];
	then
		echo "Failed"
		echo "No output was generated"
		return 0
	fi

	tmp="$(mktemp -t MEA.XXXXX)"
	tmp2="$(mktemp -t MEA.XXXXX)"
	grep -v 'Time' $INOUT_DIR/$OUT_FILE > $tmp
	grep -v 'Time' $expected_output > $tmp2
	
	diff_res=$(diff $tmp $tmp2)
	if [ -n "$diff_res" ] 
	then
        # Use sdiff if it exists in system
        if [ -z `type -t sdiff` ]; then
            diff='diff'
        else
            diff='sdiff'
        fi
        echo "> $sdiff $tmp $tmp2"
	    $diff $tmp $tmp2	
        echo "ERROR: outputs mismatch, see diff above"
        return 1
	else
        echo "ALL OK"
		rm $tmp $tmp2
        return 0
	fi 
}

models=(model_p53.txt model_MM.txt model_dimer.txt model_Hes1.txt)
for m in "${models[@]}"
do
	echo "testing $m:"
    testModel "--MEA --model=$INOUT_DIR/$m --ODEout=$OUT_FILE" "$INOUT_DIR/$m.out"
    # Check if last command failed, and exit
    if [ $? -ne 0 ]; then
        exit 1
    fi
done
