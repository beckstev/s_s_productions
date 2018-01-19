all: build/main.pdf

debug: versuchsaufbau.tex	theorie.tex main.tex diskussion.tex auswertung.tex lit.bib build/header.fmt| build
	 lualatex --output-directory=build --draft -shell-escape "&./build/header main.tex"

build/main.pdf: versuchsaufbau.tex	theorie.tex main.tex diskussion.tex auswertung.tex lit.bib ../Messdaten/tabs/*| build
			lualatex --output-directory=build main.tex

full: versuchsaufbau.tex	theorie.tex main.tex diskussion.tex auswertung.tex lit.bib | build
		lualatex --output-directory=build main.tex
		biber build/main.bcf
		lualatex --output-directory=build main.tex

build:
		mkdir -p build

clean:
		rm -rf build
.PHONY: clean all
