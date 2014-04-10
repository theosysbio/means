

all: writeup

writeup: pipeline_figures mea_speed_figure
	@echo "COMPILING LATEX REPORT!!=================================="
	make -C  writeup




mea_speed_figure:
	make -C  figure_mea_speed

pipeline_figures:
	make -C  pipeline


cleanall:
	make -C  writeup clean
	make -C  pipeline clean
