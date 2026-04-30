# Overleaf Thesis Project

This directory is the maintained LaTeX thesis package.

## Usage

1. Upload the whole `overleaf_thesis/` directory to Overleaf.
2. Set the compiler to **XeLaTeX**.
3. Compile `main.tex`.

## Structure

- `main.tex`: active Overleaf entry. It keeps only the preamble, `\xdusetup`, and ordered `\input{sections/...}` calls.
- `metadata.tex`: metadata source for title, author, student ID, major, advisor, college, and date.
- `sections/00_abstract_cn.tex` / `sections/01_abstract_en.tex`: abstract files read by the thesis class.
- `sections/02_chapter_01.tex`, `sections/04_chapter_03.tex` ... `sections/10_chapter_09.tex`: canonical thesis body sources, input by `main.tex`.
- `sections/11_acknowledgements.tex`: acknowledgement content read by the thesis class; do not add a chapter heading in this file.
- `sections/12_appendix.tex`: appendix content read by the thesis class after it switches to appendix mode.
- `sections/03_chapter_02.tex`: legacy standalone literature-review file kept only for reference and not part of the active build.
- `_inline_abstract_cn.tex` / `_inline_abstract_en.tex`: legacy abstract copies retained for compatibility; they are not used by the active build.
- `references.bib`: BibTeX database used by `gbt7714-numerical`.
- `assets/figures/`: figures and table images referenced by the thesis.

## Notes

- Use `main.tex` as the final compilation entry.
- Edit thesis body content only in `sections/*.tex`; no chapter content should be duplicated in `main.tex`.
- Backmatter is handled by `xduugthesis.cls` through `info / acknowledgements`, `info / bib-resource`, and `info / appendix`.
- Compile from this directory with `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`.
