#!/bin/bash

##  The expected out files are named`MODEL_FILENAME.txt.out`
INOUT_DIR="../Inoutput"
MODEL_ANSWERS_DIR="../Inoutput/model_answers/"

buildRefs="false"
max_mom=2
test_simulation="true"

generateReferenceResults(){
    parameters=$1
    expected_output=$2
    echo "> python runprogram.py $parameters --ODEout=$out_file"
	python runprogram.py $parameters --ODEout=$out_file
	echo "> mv $INOUT_DIR/$out_file  $expected_output"
	mv $INOUT_DIR/$out_file  $expected_output

}



testModel(){
    parameters=$1
    out_file=$2
    expected_output=$3

    echo "> python runprogram.py $parameters"	
	time python runprogram.py $parameters

    if [ ! -f $INOUT_DIR/$out_file ];
	then
		echo "Failed"
		echo "No output was generated"
		return 1
	fi

	tmp="$(mktemp -t MEA.ACTUAL.XXXXX)"
	tmp2="$(mktemp -t MEA.EXPECTED.XXXXX)"
	cat $INOUT_DIR/$out_file | grep -v "Time taken" | grep -v "Input file:" > $tmp
    cat $INOUT_DIR/$expected_output | grep -v "Time taken" | grep -v "Input file:" > $tmp2
    
    # Not removing this may make the next test appear to be failing with output
    # mismatch even though it did not produce any
    rm $INOUT_DIR/$out_file
	
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
# MEA tests
for i in $(seq 2 $max_mom)
do
	for m in "${models[@]}"
		do
		if [ $buildRefs == "true" ]; then
			generateReferenceResults "--MEA --nMom=$i --model=$INOUT_DIR/$m" "$MODEL_ANSWERS_DIR/MEA$i/$m.out"
		else
			echo "testing $m:"
			testModel "--MEA --nMom=$i --model=$INOUT_DIR/$m --ODEout=ODEout.tmp" "ODEout.tmp" "$MODEL_ANSWERS_DIR/MEA$i/$m.out"
			
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
		testModel "--LNA --model=$INOUT_DIR/$m --ODEout=ODEout.tmp" "ODEout.tmp" "$MODEL_ANSWERS_DIR/LNA/$m.out"
		# Check if last command failed, and exit
		if [ $? -ne 0 ]; then
			exit 1
		fi
	fi
done


# simulation tests this has the potential of being awfully broken due to floating point rounding
simtest_models=("MM" "p53")

# Let's find the sundials library on your computer
sundials_parameters=""
if [ -f "/usr/local/lib/libsundials_cvode.a" ];
then
    # This is where sundials is on Mac OS X if installed from brew
    sundials_parameters="--sd2=/usr/local/lib/ --sd1=/usr/local/include/"
else
    echo "ERROR: Cannot run simulation tests as cannot file sundials library"
    exit 1
fi
if [ $test_simulation == "true" ]; then
    for m in "${simtest_models[@]}" 
    do
        echo "Testing model $m:"
        testModel "--MEA --nMom=3 --model=../Inoutput/model_$m.txt --compile $sundials_parameters --timeparam=../Inoutput/param_$m.txt --sim --simout=simout_$m.txt --ODEout=ODEout.tmp" "simout_$m.txt" "$MODEL_ANSWERS_DIR/sim/simout_$m.txt"

        if [ $? -ne 0 ]; then
            exit 1
        fi
    done
fi
test_inference="true"
inference_models=("dimer")
if [ $test_inference == "true" ]; then
   for m in "${inference_models[@]}"
   do
       echo "Testing model $m:"
       testModel "--MEA --model=model_$m.txt --ODEout=ODEout.tmp --compile --library=$m.tmp --timeparam=param_$m.txt --infer --data=data_${m}_x40.txt --inferfile=inferout.tmp $sundials_parameters" "inferout.tmp" "$MODEL_ANSWERS_DIR/infer/infer_$m.txt"

       if [ $? -ne 0 ]; then
           exit 1
       fi
   done
fi
