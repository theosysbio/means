OUT="bench.tmp"
PYTHON_SCRIPT="/home/quentin/Desktop/benchmark.py"
tags="no_simplify_and_cache_diff use_xreplace only_necessary_moms"

# We generate data
echo "method, n_ODEs, t" > $OUT;
MAX_ORDER=4
for t in $tags;
	do
	git checkout $t
	sleep 1
	python $PYTHON_SCRIPT $t $MAX_ORDER >> $OUT;
	done

# we plot with R ggplot
rcom="	library(ggplot2);
		pdf('bench.pdf');
		df <- read.table('$OUT',h=T,sep=',');
		df\$max_order <- as.numeric(as.factor(df\$n_ODEs));
		ggplot(data=df, aes(x=max_order, y=log2(t), group=method, colour=method, shape=method)) + 
		geom_line() + geom_point() + geom_line(size=1.5) +
		geom_point(size=3, fill='white');
		dev.off()"

echo $rcom | R --vanilla  
