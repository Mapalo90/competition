# Technical report

`main.tex` is written in the competition's official Overleaf template (ACM `acmart` class, `sigconf`).
`preamble.tex` and the first two entries of `main.bib` reproduce the template's files; the compiled layout
matches the template PDF word for word. `report.pdf` is the compiled report.

To compile: upload this folder to Overleaf (New Project > Upload Project) and compile with pdfLaTeX, or locally with
`latexmk -pdf main.tex` (acmart 2.14, the version in TeX Live 2025).

Figures (`figures/`): the pipeline and the examples use fold-0 images with out-of-fold predictions of the fold-0
model; the IoU / Dice / one-to-many distributions and the PQ-by-annotators plot come from the notebook's saved
outputs (`pair_stats.npz`, `per_image_counts.csv`); the sensitivity plot from the post-processing search grid.
