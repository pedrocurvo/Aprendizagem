%.pdf : %.tex
	echo "Compiling LaTeX"
	pdflatex -output-directory=bin $^
	mv bin/*.pdf $@