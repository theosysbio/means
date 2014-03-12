MYDIR := $(shell basename `pwd`)
OUT_DATA := ../$(MYDIR).pickle

all : $(OUT_DATA)

$(OUT_DATA): script.py Makefile
	python $< $@

	
clean:
	rm -f *.pikle  .pdf *.log *.csv *.pyc $(OUT_DATA)

