#!/bin/bash

##  The expected out files are named`MODEL_FILENAME.txt.out`
INOUT_DIR="../Inoutput/"
OUT_FILE="ODEout.tmp"



testModel(){
	model=$1
	good_result=$INOUT_DIR/$model.out

	python runprogram.py --MEA --model=$INOUT_DIR/$model --ODEout=$OUT_FILE
	
	if [ ! -f $INOUT_DIR/$OUT_FILE ];
	then
		echo "$model FAILED!!"
		echo "No output was generated"
		return $FALSE
	fi

	tmp="$(mktemp -t MEA.XXXXX)"
	tmp2="$(mktemp -t MEA.XXXXX)"
	grep -v 'Time' $INOUT_DIR/$OUT_FILE > $tmp
	grep -v 'Time' $good_result > $tmp2
	
	diff_res=$(diff $tmp $tmp2)
	if [ "$dif_res" != "" ] 
	then
		echo "$model FAILED!!"
		return $FALSE
	else
		echo "$model OK"
		return $TRUE
	fi 
	
	

	rm $tmp $tmp2
}



models=(model_p53.txt model_MM.txt model_dimer.txt model_Hes1.txt)
for m in "${models[@]}"
do
	echo "testing $m:"
	testModel $m
done
