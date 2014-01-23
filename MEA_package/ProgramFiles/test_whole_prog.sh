#!/bin/bash

##  The expected out files are named`MODEL_FILENAME.txt.out`
INOUT_DIR="../Inoutput"
MODEL_ANSWERS_DIR="../Inoutput/model_answers/"
OUT_FILE="ODEout.tmp"

generateReferenceResults(){
    parameters=$1
    expected_output=$2
    echo "> python runprogram.py $parameters --ODEout=$OUT_FILE"
	python runprogram.py $parameters --ODEout=$OUT_FILE
	echo "> mv $INOUT_DIR/$OUT_FILE  $expected_output"
	mv $INOUT_DIR/$OUT_FILE  $expected_output

}



testModel(){
    parameters=$1
    expected_output=$2

    echo "> python runprogram.py $parameters --ODEout=$OUT_FILE"	
	time python runprogram.py $parameters --ODEout=$OUT_FILE

    if [ ! -f $INOUT_DIR/$OUT_FILE ];
	then
		echo "Failed"
		echo "No output was generated"
		return 1
	fi

	tmp="$(mktemp -t MEA.XXXXX)"
	tmp2="$(mktemp -t MEA.XXXXX)"
	grep -v 'Time' $INOUT_DIR/$OUT_FILE > $tmp
	grep -v 'Time' $expected_output > $tmp2
    
    # Not removing this may make the next test appear to be failing with output
    # mismatch even though it did not produce any
    rm $INOUT_DIR/$OUT_FILE
	
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

buildRefs="true"
max_mom=3

models=(model_p53.txt model_MM.txt model_dimer.txt model_Hes1.txt)
# MEA tests
for i in $(seq 2 $max_mom)
do
	for m in "${models[@]}"
		do
		if [ $buildRefs == "true" ]; then
			generateReferenceResults "--MEA --nMom=$i --model=$INOUT_DIR/$m" "$MODEL_ANSWERS_DIR/MEA$i/$m.out"
		else
			echo "testing $m:"
			testModel "--MEA --nMom=$i --model=$INOUT_DIR/$m" "$MODEL_ANSWERS_DIR/MEA$i/$m.out"
			
			# Check if last command failed, and exit
			if [ $? -ne 0 ]; then
				exit 1
			fi
		fi
	done
done

# LNA tests
for m in "${models[@]}"
do	
	if [ $buildRefs == "true" ]; then
		generateReferenceResults "--LNA --model=$INOUT_DIR/$m" "$MODEL_ANSWERS_DIR/LNA/$m.out"
	else
		echo "testing $m:"
		testModel "--LNA --model=$INOUT_DIR/$m" "$MODEL_ANSWERS_DIR/LNA/$m.out"
		# Check if last command failed, and exit
		if [ $? -ne 0 ]; then
			exit 1
		fi
	fi
done
