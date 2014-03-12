toclean := $(shell ls figure_*/Makefile |sed s/Makefile//g)
toclean += $(shell ls data_*/Makefile |sed s/Makefile//g)

all : figures

clean:
	rm -f *.pickle

cleanall: clean
	for i in $(toclean); do $(MAKE) -C $$i clean; done


#figures: _figure_benchmark
figures: _figure_closure_and_max_order_hes1 _figure_closure_and_max_order_p53 _figure_closure_and_max_order_dimer _figure_closure_and_max_order_lotka_volterra
#data: _data_benchmark


_data_closure_and_max_order_hes1:
	@$(MAKE) -C data_closure_and_max_order_hes1

_figure_closure_and_max_order_hes1: _data_closure_and_max_order_hes1
	@$(MAKE) -C figure_closure_and_max_order_hes1


_data_closure_and_max_order_p53:
	@$(MAKE) -C data_closure_and_max_order_p53

_figure_closure_and_max_order_p53: _data_closure_and_max_order_p53
	@$(MAKE) -C figure_closure_and_max_order_p53


_data_closure_and_max_order_dimer:
	@$(MAKE) -C data_closure_and_max_order_dimer

_figure_closure_and_max_order_dimer: _data_closure_and_max_order_dimer
	@$(MAKE) -C figure_closure_and_max_order_dimer


_data_closure_and_max_order_lotka_volterra:
	@$(MAKE) -C data_closure_and_max_order_lotka_volterra

_figure_closure_and_max_order_lotka_volterra: _data_closure_and_max_order_lotka_volterra
	@$(MAKE) -C figure_closure_and_max_order_lotka_volterra


_figure_benchmark: _data_benchmark
	@$(MAKE) -C figure_benchmark

_data_benchmark:
	@$(MAKE) -C data_benchmark
