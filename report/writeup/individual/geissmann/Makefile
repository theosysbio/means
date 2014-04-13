all: report.pdf

ALL_TEX := $(shell ls *.tex)

report.pdf: $(ALL_TEX) ../../report.bib Makefile
	echo "Generating pdf"
	latexmk report.tex
	
clean:
	latexmk -C
