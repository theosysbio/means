toclean := $(shell ls figure_*/Makefile |sed s/Makefile//g)
#toclean += $(shell ls data_*/Makefile |sed s/Makefile//g)

all : figures

clean:
	rm -f *.pickle

cleanall: clean
	for i in $(toclean); do $(MAKE) -C $$i clean; done

figures: _figure_benchmark

_figure_benchmark:
	@$(MAKE) -C figure_benchmark

