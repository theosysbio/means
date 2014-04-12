STATIC_HTML: ../*.ipynb ../images
	ipython nbconvert html ../*.ipynb
	cp -R ../images .
clean:
	rm *.html
	rm -rf images

all: STATIC_HTML

