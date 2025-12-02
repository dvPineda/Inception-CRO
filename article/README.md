# Article: Inception-CRO

This directory contains the LaTeX source files for the scientific article on Inception-CRO algorithm.

## Structure

```
article/
├── latex/
│   ├── main.tex                    # Main LaTeX file (Elsevier template)
│   ├── sections/                   # Individual sections
│   │   ├── introduction.tex        # Introduction section
│   │   ├── background.tex          # Background and related work
│   │   ├── methodology.tex         # Methodology and algorithm description
│   │   ├── experiments.tex         # Experimental setup
│   │   ├── results.tex             # Results and discussion
│   │   └── conclusion.tex          # Conclusion and future work
│   ├── figures/                    # Figures and plots
│   ├── tables/                     # Table files (if separate)
│   └── bibliography/               # Bibliography files
│       └── references.bib          # BibTeX references
└── README.md                       # This file
```

## Compilation

To compile the article:

1. Ensure you have a complete LaTeX distribution installed (e.g., TeX Live, MiKTeX)
2. Navigate to the `latex/` directory
3. Run the following commands:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk for automatic compilation:

```bash
latexmk -pdf main.tex
```

## Template Information

The article uses the Elsevier `elsarticle` document class, which is suitable for submission to Elsevier journals. The template includes:

- Proper formatting for scientific articles
- Support for algorithms, figures, tables, and equations
- Bibliography management with BibTeX
- Cross-referencing capabilities

## Required Packages

The template uses the following LaTeX packages:
- `amssymb`, `amsmath` - Mathematical symbols and environments
- `algorithm`, `algorithmic` - Algorithm typesetting
- `booktabs` - Professional table formatting
- `multirow` - Multiple row cells in tables
- `subcaption` - Subfigures and subcaptions
- `graphicx` - Graphics inclusion
- `url` - URL formatting
- `hyperref` - Hyperlinks and cross-references

## Adding Figures

1. Place figure files (PDF, PNG, EPS) in the `figures/` directory
2. Reference them in the LaTeX source using relative paths:
   ```latex
   \includegraphics[width=0.8\textwidth]{figures/your_figure.pdf}
   ```

## Adding References

1. Add new references to `bibliography/references.bib` in BibTeX format
2. Cite them in the text using `\cite{reference_key}`

## Notes

- The main article title and authors should be updated in `main.tex`
- The journal name can be specified using `\journal{Journal Name}`
- The abstract and keywords should be customized for your specific research
- Figure files are not included in this template - you'll need to generate and add them based on your experimental results
