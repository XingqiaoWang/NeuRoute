# Paper Assets

This directory contains the paper artifacts for NeuRoute.

- `NeuRoute_VLDB2026.pdf`: main paper PDF.
- `Appendix.pdf`: appendix PDF.
- `overleaf/`: LaTeX source bundle and paper figures from Overleaf.

The Overleaf bundle preserves several historical internal names, including `Autohash` and `SPHash`, because those names appear in older source paths and figure filenames. The public system name is **NeuRoute**.

To compile the main manuscript locally, run from `paper/overleaf`:

```bash
pdflatex Autohash/Autohash_VLDB2026.tex
bibtex Autohash_VLDB2026
pdflatex Autohash/Autohash_VLDB2026.tex
pdflatex Autohash/Autohash_VLDB2026.tex
```

`references.bib` is duplicated at the Overleaf root from `Autohash/references.bib` to make local BibTeX runs simpler.
