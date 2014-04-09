all: report.pdf

ALL_TEX := $(shell ls *.tex)

tutorials: generate_tutorials.py ../../prototypes/examples/Tutorial/*.ipynb
	python generate_tutorials.py ../../prototypes/examples/Tutorial/

report.pdf: $(ALL_TEX) report.bib Makefile
	echo "Generating pdf"
	latexmk report.tex
	
clean:
	latexmk -C
