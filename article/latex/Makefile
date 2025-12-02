# Makefile for compiling the Inception-CRO article

# Main document name (without .tex extension)
MAIN = main

# Default target
all: $(MAIN).pdf

# Compile the PDF
$(MAIN).pdf: $(MAIN).tex sections/*.tex bibliography/references.bib
	pdflatex $(MAIN).tex
	bibtex $(MAIN)
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

# Quick compile (no bibliography update)
quick: $(MAIN).tex sections/*.tex
	pdflatex $(MAIN).tex

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Clean everything including PDF
cleanall: clean
	rm -f $(MAIN).pdf

# Continuous compilation with latexmk
watch:
	latexmk -pdf -pvc $(MAIN).tex

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Compile the complete document (default)"
	@echo "  quick    - Quick compile without bibliography update"
	@echo "  clean    - Remove auxiliary files"
	@echo "  cleanall - Remove all generated files including PDF"
	@echo "  watch    - Continuous compilation with latexmk"
	@echo "  help     - Show this help message"

.PHONY: all quick clean cleanall watch help
