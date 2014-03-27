all: report.pdf

ALL_TEX := $(shell ls *.tex)


report.pdf: $(ALL_TEX) report.bib
	echo "Generating pdf"
	xelatex  --shell-escape $(basename $@).tex
	bibtex $(basename $@)
	xelatex  --shell-escape $(basename $@).tex 
	xelatex  --shell-escape $(basename $@) 


clean:
	rm -f *{.log,.dvi,.aux,.out,.bbl,.pdf,.dvi,.blg,.toc,.pyg}
