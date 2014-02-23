tags="no_simplify_and_cache_diff use_xreplace only_necessary_moms use_quick_solve"
OUT="bench.tmp"

function run_python_script(){
	mo=$1
	PYTHON_SCRIPT="import time;
from means.approximation.mea import MomentExpansionApproximation;
from means.examples.sample_models import MODEL_P53;
t0 = time.time();
pb = MomentExpansionApproximation(MODEL_P53, $mo, 'log-normal').run();
print '{0}, {1}'.format(pb.number_of_equations, time.time() - t0);"
	
	echo $(python -c "$PYTHON_SCRIPT")
}


# We generate data

MAX_ORDER=3
# csv header
echo "method, n_ODEs, t" > $OUT
for mo in $(seq 2 $MAX_ORDER)
	do
	for t in $tags;
		do
		git checkout $t 2> /dev/null
		echo "tag: $t, max_order: $mo"
		sleep 3
		res=$(run_python_script $mo)
		# One row
		echo $t, $res >> $OUT
	done
done

# When data is generated, we plot
# with R ggplot:
rcom="	library(ggplot2);
		pdf('bench.pdf', width=10, height=6);
		df <- read.table('$OUT',h=T,sep=',');
		df\$max_order <- as.numeric(as.factor(df\$n_ODEs));
		ggplot(data=df, aes(x=max_order, y=log2(t), group=method, colour=method, shape=method)) + 
		geom_line() + geom_point() + geom_line(size=1.5) +
		geom_point(size=3, fill='white');
		dev.off()"

echo $rcom | R --vanilla  
