
all: report.pdf
		
ALL_TEX := $(shell ls *.tex)
report.pdf: $(ALL_TEX) report.bib
	echo "Generating pdf"
	latex $(basename $@).tex
	bibtex $(basename $@)
	latex $(basename $@).tex
	latex $(basename $@)
	dvipdf $(basename $@)

clean:
	rm -f *{.log,.dvi,.aux,.out,.bbl,.pdf,.dvi, blg}
