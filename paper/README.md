# Paper Assets

This directory contains the paper artifacts for NeuRoute.

- `NeuRoute_arXiv.pdf`: main arXiv/preprint paper PDF.
- `Appendix.pdf`: appendix PDF.
- `overleaf/`: LaTeX source bundle and paper figures from Overleaf.

The Overleaf bundle preserves several historical internal names, including `Autohash` and `SPHash`, because those names appear in older source paths and figure filenames. The public system name is **NeuRoute**. This is a preprint/arXiv manuscript bundle, not a conference proceedings version.

To compile the main manuscript locally, run from `paper/overleaf`:

```bash
pdflatex Autohash/NeuRoute_arXiv.tex
bibtex NeuRoute_arXiv
pdflatex Autohash/NeuRoute_arXiv.tex
pdflatex Autohash/NeuRoute_arXiv.tex
```

`references.bib` is duplicated at the Overleaf root from `Autohash/references.bib` to make local BibTeX runs simpler.
