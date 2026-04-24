# Overleaf Thesis Project

This directory was generated from `论文初稿.md`.

## Usage

1. Upload the whole `overleaf_thesis/` directory to Overleaf.
2. Set the compiler to **XeLaTeX**.
3. Edit `metadata.tex` to fill in your:
   - author name
   - student ID
   - major
   - advisor
   - college
   - date
4. Compile `main.tex`.

## Structure

- `main.tex`: main entry for Overleaf
- `metadata.tex`: thesis cover metadata
- `sections/*.tex`: generated thesis content split by chapter

## Notes

- This project keeps the original thesis content and converts it into a basic LaTeX layout.
- Reference items are preserved as a chapter instead of being converted into BibTeX automatically.
- If you later update the markdown draft, rerun:

```bash
python scripts/convert_thesis_to_latex.py
```
