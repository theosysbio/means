toclean := $(shell ls figure_*/Makefile |sed s/Makefile//g)
toclean += $(shell ls data_*/Makefile |sed s/Makefile//g)

all : figures

clean:
	rm -f *.pickle

cleanall: clean
	for i in $(toclean); do $(MAKE) -C $$i clean; done


#figures: _figure_benchmark
figures: _figure_best_order_mom
#data: _data_benchmark


_figure_best_order_mom: _data_best_order_mom
	@$(MAKE) -C figure_best_order_mom


_figure_benchmark: _data_benchmark
	@$(MAKE) -C figure_benchmark

_data_benchmark:
	@$(MAKE) -C data_benchmark

_data_best_order_mom:
	@$(MAKE) -C data_best_order_mom
