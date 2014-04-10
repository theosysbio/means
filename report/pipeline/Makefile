ALL_FIGURE_TASKS := figure_traject_plot_exple.py figure_closure_and_max_order.py

ALL_FIGURE_TO_CLEAN := task-output/FigureP53Simple/FigureP53Simple-pdf-7.pdf task-output/FigureP53Simple/FigureP53Simple-pdf-7.pdf 

all: 
	for i in $(ALL_FIGURE_TASKS); do \
		python2 $$i; \
		done;

clean:
	rm -f $(ALL_FIGURE_TO_CLEAN)
