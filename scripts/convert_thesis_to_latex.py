#!/usr/bin/env python3
"""
Convert the thesis markdown draft into an Overleaf-ready LaTeX project.

Default source:
    论文初稿.md

Default output:
    overleaf_thesis/

The generated project uses XeLaTeX + ctexrep and splits the thesis into
multiple `sections/*.tex` files so it can be uploaded to Overleaf directly.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "论文初稿.md"
DEFAULT_OUTPUT = ROOT / "overleaf_thesis"


CHAPTER_NAME_RE = re.compile(r"^第[一二三四五六七八九十百零]+章\s*")
SECTION_NUM_RE = re.compile(r"^\d+(?:\.\d+)*\s*")
ORDERED_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")
CJK_ORDERED_RE = re.compile(r"^\s*（(\d+)）\s*(.*)$")
UNORDERED_RE = re.compile(r"^\s*-\s+(.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
URL_RE = re.compile(r"https?://[^\s）)\]}>]+")
INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)")


@dataclass
class SectionFile:
    kind: str
    title: str
    filename: str
    content: list[str]


def normalize_title(title: str) -> str:
    title = title.strip()
    title = CHAPTER_NAME_RE.sub("", title)
    title = title.removeprefix("附：").strip()
    return title


def restore_placeholders(text: str, registry: list[str]) -> str:
    for index, value in enumerate(registry):
        text = text.replace(f"LATEXPLACEHOLDERX{index}X", value)
    return text


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def convert_inline(text: str) -> str:
    registry: list[str] = []
    working = text

    def add(fragment: str) -> str:
        token = f"LATEXPLACEHOLDERX{len(registry)}X"
        registry.append(fragment)
        return token

    def replace_math(match: re.Match[str]) -> str:
        return add(r"\(" + match.group(1) + r"\)")

    def replace_code(match: re.Match[str]) -> str:
        code = escape_latex(match.group(1))
        return add(rf"\texttt{{{code}}}")

    def replace_url(match: re.Match[str]) -> str:
        return add(rf"\url{{{match.group(0)}}}")

    def replace_bold(text_in: str) -> str:
        pattern = re.compile(r"\*\*(.+?)\*\*")
        while True:
            match = pattern.search(text_in)
            if match is None:
                return text_in
            replacement = add(rf"\textbf{{{convert_inline(match.group(1))}}}")
            text_in = text_in[: match.start()] + replacement + text_in[match.end() :]

    def replace_italic(text_in: str) -> str:
        pattern = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
        while True:
            match = pattern.search(text_in)
            if match is None:
                return text_in
            replacement = add(rf"\emph{{{convert_inline(match.group(1))}}}")
            text_in = text_in[: match.start()] + replacement + text_in[match.end() :]

    working = INLINE_MATH_RE.sub(replace_math, working)
    working = re.sub(r"`([^`]+)`", replace_code, working)
    working = URL_RE.sub(replace_url, working)
    working = replace_bold(working)
    working = replace_italic(working)
    working = escape_latex(working)
    working = restore_placeholders(working, registry)
    return working


def strip_heading_marker(text: str, level: int) -> str:
    cleaned = text.strip()
    if level == 2 and cleaned.startswith("第") and "章" in cleaned:
        return normalize_title(cleaned)
    if level >= 3:
        return SECTION_NUM_RE.sub("", cleaned).strip()
    return cleaned


def parse_blocks(lines: list[str], start: int = 0, stop: int | None = None) -> tuple[list[str], int]:
    """Render markdown lines into LaTeX lines until `stop` or end."""
    out: list[str] = []
    i = start
    limit = len(lines) if stop is None else min(stop, len(lines))

    def is_continuation(candidate: str) -> bool:
        return bool(candidate) and bool(re.match(r"^\s{2,}\S", candidate))

    def peek_nonempty(index: int) -> str | None:
        while index < limit:
            stripped = lines[index].strip()
            if stripped:
                return stripped
            index += 1
        return None

    while i < limit:
        raw = lines[i].rstrip("\n")
        stripped = raw.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            i += 1
            continue

        if stripped.startswith(r"\["):
            math_lines = [raw]
            i += 1
            while i < limit:
                math_lines.append(lines[i].rstrip("\n"))
                if lines[i].strip().endswith(r"\]"):
                    i += 1
                    break
                i += 1
            out.extend(math_lines)
            out.append("")
            continue

        heading = HEADING_RE.match(stripped)
        if heading:
            level = len(heading.group(1))
            title = strip_heading_marker(heading.group(2), level)
            if level == 3:
                out.append(rf"\section{{{convert_inline(title)}}}")
            elif level == 4:
                out.append(rf"\subsection{{{convert_inline(title)}}}")
            elif level == 5:
                out.append(rf"\subsubsection{{{convert_inline(title)}}}")
            else:
                out.append(rf"\paragraph{{{convert_inline(title)}}}")
            out.append("")
            i += 1
            continue

        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while i < limit and lines[i].strip().startswith(">"):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[i].rstrip("\n")))
                i += 1
            quote_body, _ = parse_blocks(quote_lines, 0, len(quote_lines))
            out.append(r"\begin{quote}")
            out.extend(quote_body)
            out.append(r"\end{quote}")
            out.append("")
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < limit and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            out.extend(render_table(table_lines))
            out.append("")
            continue

        if ORDERED_RE.match(stripped) or CJK_ORDERED_RE.match(stripped):
            list_lines = []
            list_kind = "ordered"
            while i < limit:
                candidate = lines[i].rstrip("\n")
                cstripped = candidate.strip()
                if not cstripped:
                    next_nonempty = peek_nonempty(i + 1)
                    if next_nonempty and (
                        ORDERED_RE.match(next_nonempty)
                        or CJK_ORDERED_RE.match(next_nonempty)
                    ):
                        i += 1
                        continue
                    break
                if not (
                    ORDERED_RE.match(cstripped)
                    or CJK_ORDERED_RE.match(cstripped)
                    or candidate.startswith("\t")
                    or is_continuation(candidate)
                ):
                    break
                list_lines.append(candidate)
                i += 1
            out.extend(render_list(list_lines, list_kind))
            out.append("")
            continue

        if UNORDERED_RE.match(stripped):
            list_lines = []
            while i < limit:
                candidate = lines[i].rstrip("\n")
                cstripped = candidate.strip()
                if not cstripped:
                    next_nonempty = peek_nonempty(i + 1)
                    if next_nonempty and UNORDERED_RE.match(next_nonempty):
                        i += 1
                        continue
                    break
                if not (
                    UNORDERED_RE.match(cstripped)
                    or candidate.startswith("\t")
                    or is_continuation(candidate)
                ):
                    break
                list_lines.append(candidate)
                i += 1
            out.extend(render_list(list_lines, "unordered"))
            out.append("")
            continue

        paragraph_lines = [stripped]
        i += 1
        while i < limit:
            candidate = lines[i].strip()
            if (
                not candidate
                or candidate == "---"
                or candidate.startswith(">")
                or candidate.startswith("|")
                or HEADING_RE.match(candidate)
                or candidate.startswith(r"\[")
                or ORDERED_RE.match(candidate)
                or CJK_ORDERED_RE.match(candidate)
                or UNORDERED_RE.match(candidate)
            ):
                break
            paragraph_lines.append(candidate)
            i += 1
        out.append(convert_inline(" ".join(paragraph_lines)))
        out.append("")

    return out, i


def render_list(lines: list[str], kind: str) -> list[str]:
    items: list[list[str]] = []
    current: list[str] | None = None

    for line in lines:
        stripped = line.strip()
        ordered_match = ORDERED_RE.match(stripped) or CJK_ORDERED_RE.match(stripped)
        unordered_match = UNORDERED_RE.match(stripped)

        if kind == "ordered" and ordered_match:
            if current is not None:
                items.append(current)
            current = [ordered_match.group(2)]
            continue
        if kind == "unordered" and unordered_match:
            if current is not None:
                items.append(current)
            current = [unordered_match.group(1)]
            continue

        if current is None:
            current = [stripped]
        else:
            current.append(stripped)

    if current is not None:
        items.append(current)

    env = "enumerate" if kind == "ordered" else "itemize"
    options = "[leftmargin=2em]" if env == "itemize" else "[leftmargin=2.5em]"
    out = [rf"\begin{{{env}}}{options}"]
    for item_lines in items:
        rendered_parts = [convert_inline(part) for part in item_lines if part]
        if not rendered_parts:
            out.append(r"\item")
        elif len(rendered_parts) == 1:
            out.append(rf"\item {rendered_parts[0]}")
        else:
            joined = r" \\" + "\n" + "  "
            out.append(r"\item " + joined.join(rendered_parts))
    out.append(rf"\end{{{env}}}")
    return out


def split_table_row(row: str) -> list[str]:
    stripped = row.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def render_table(lines: list[str]) -> list[str]:
    rows = [split_table_row(line) for line in lines]
    if len(rows) < 2:
        return [convert_inline(" ".join(rows[0]))] if rows else []

    header = rows[0]
    data_rows = rows[2:] if len(rows) >= 2 else []
    col_count = len(header)
    if col_count == 0:
        return []

    align = ["l"] + ["r"] * (col_count - 1)
    spec = "@{}" + "".join(align) + "@{}"

    out = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(convert_inline(cell) for cell in header) + r" \\",
        r"\midrule",
    ]
    for row in data_rows:
        padded = row + [""] * (col_count - len(row))
        out.append(" & ".join(convert_inline(cell) for cell in padded[:col_count]) + r" \\")
    out.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return out


def build_sections(lines: list[str]) -> tuple[str, list[SectionFile]]:
    title = ""
    sections: list[SectionFile] = []
    current: SectionFile | None = None

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        stripped = raw.strip()
        heading = HEADING_RE.match(stripped)

        if heading:
            level = len(heading.group(1))
            heading_text = heading.group(2).strip()
            if level == 1:
                title = heading_text
                i += 1
                continue

            if level == 2:
                normalized = normalize_title(heading_text)
                if heading_text == "摘要":
                    current = SectionFile("abstract_cn", "摘要", "00_abstract_cn.tex", [])
                    sections.append(current)
                elif heading_text == "Abstract":
                    current = SectionFile("abstract_en", "Abstract", "01_abstract_en.tex", [])
                    sections.append(current)
                elif heading_text.startswith("参考文献"):
                    current = SectionFile("references", normalized, "11_references.tex", [])
                    sections.append(current)
                elif heading_text.startswith("附："):
                    current = SectionFile("appendix", normalized, "12_appendix.tex", [])
                    sections.append(current)
                else:
                    chapter_number = len([item for item in sections if item.kind == "chapter"]) + 1
                    current = SectionFile(
                        "chapter",
                        normalized,
                        f"{chapter_number + 1:02d}_chapter_{chapter_number:02d}.tex",
                        [],
                    )
                    sections.append(current)
                i += 1
                continue

        if current is None:
            i += 1
            continue

        current.content.append(raw)
        i += 1

    return title, sections


def render_section(section: SectionFile) -> str:
    body, _ = parse_blocks(section.content)
    title = convert_inline(section.title)

    if section.kind == "abstract_cn":
        header = [
            r"\chapter*{摘\quad 要}",
            r"\addcontentsline{toc}{chapter}{摘要}",
            "",
        ]
    elif section.kind == "abstract_en":
        header = [
            r"\chapter*{Abstract}",
            r"\addcontentsline{toc}{chapter}{Abstract}",
            "",
        ]
    elif section.kind == "references":
        header = [
            rf"\chapter{{{title}}}",
            "",
        ]
    elif section.kind == "appendix":
        header = [
            rf"\chapter{{{title}}}",
            "",
        ]
    else:
        header = [
            rf"\chapter{{{title}}}",
            "",
        ]
    return "\n".join(header + body).rstrip() + "\n"


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def generate_metadata() -> str:
    return r"""\newcommand{\thesistitle}{基于深度估计与多任务学习的高速公路团雾监测方法研究}
\newcommand{\thesisauthor}{请填写姓名}
\newcommand{\thesisstudentid}{请填写学号}
\newcommand{\thesismajor}{请填写专业}
\newcommand{\thesisadvisor}{请填写指导教师}
\newcommand{\thesiscollege}{请填写学院}
\newcommand{\thesisdate}{\today}
"""


def generate_main_tex(project_title: str, sections: list[SectionFile]) -> str:
    frontmatter_files = [item for item in sections if item.kind in {"abstract_cn", "abstract_en"}]
    mainmatter_files = [item for item in sections if item.kind == "chapter"]
    references_files = [item for item in sections if item.kind == "references"]
    appendix_files = [item for item in sections if item.kind == "appendix"]

    def inputs(items: Iterable[SectionFile]) -> str:
        return "\n".join(rf"\input{{sections/{item.filename}}}" for item in items)

    return rf"""\documentclass[UTF8,openany,oneside]{{ctexbook}}
\usepackage[a4paper,margin=2.6cm]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{longtable}}
\usepackage{{enumitem}}
\usepackage{{hyperref}}
\usepackage{{xurl}}
\usepackage{{setspace}}
\usepackage{{fancyhdr}}
\usepackage{{titlesec}}
\usepackage{{indentfirst}}

\input{{metadata.tex}}

\hypersetup{{
  colorlinks=true,
  linkcolor=blue,
  urlcolor=blue,
  citecolor=blue,
  pdftitle={{{convert_inline(project_title)}}},
  pdfauthor={{\thesisauthor}}
}}

\setlength{{\parindent}}{{2em}}
\setlength{{\parskip}}{{0.35em}}
\onehalfspacing
\setcounter{{tocdepth}}{{2}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyfoot[C]{{\thepage}}
\renewcommand{{\headrulewidth}}{{0pt}}
\titleformat{{\chapter}}[hang]{{\centering\bfseries\zihao{{3}}}}{{第\thechapter 章}}{{1em}}{{}}
\titleformat{{\section}}{{\bfseries\zihao{{4}}}}{{\thesection}}{{1em}}{{}}
\titleformat{{\subsection}}{{\bfseries}}{{\thesubsection}}{{1em}}{{}}

\begin{{document}}

\begin{{titlepage}}
  \centering
  {{\zihao{{2}}\bfseries \thesistitle\par}}
  \vspace{{2.5cm}}
  \begin{{tabular}}{{rl}}
    学院： & \thesiscollege \\
    专业： & \thesismajor \\
    姓名： & \thesisauthor \\
    学号： & \thesisstudentid \\
    指导教师： & \thesisadvisor \\
    日期： & \thesisdate \\
  \end{{tabular}}
  \vfill
  {{\Large 本文件由 Markdown 草稿自动转换为 LaTeX，适合导入 Overleaf。\par}}
\end{{titlepage}}

\frontmatter
{inputs(frontmatter_files)}
\tableofcontents
\clearpage

\mainmatter
{inputs(mainmatter_files)}

{inputs(references_files)}
\appendix
{inputs(appendix_files)}

\end{{document}}
"""


def generate_readme(source_name: str) -> str:
    return f"""# Overleaf Thesis Project

This directory was generated from `{source_name}`.

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
"""


def main():
    parser = argparse.ArgumentParser(description="Convert thesis markdown to LaTeX.")
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Source markdown file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Output Overleaf project directory.",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source markdown not found: {source}")

    lines = source.read_text(encoding="utf-8").splitlines()
    title, sections = build_sections(lines)
    if not title:
        raise RuntimeError("Failed to find thesis title from the markdown source.")
    if not sections:
        raise RuntimeError("Failed to parse any top-level thesis sections.")

    write_file(output_dir / "metadata.tex", generate_metadata())
    write_file(output_dir / "README.md", generate_readme(source.name))
    write_file(output_dir / "main.tex", generate_main_tex(title, sections))
    for section in sections:
        write_file(output_dir / "sections" / section.filename, render_section(section))

    print(f"Generated Overleaf project: {output_dir}")
    print(f"Source markdown: {source}")
    print(f"Sections written: {len(sections)}")


if __name__ == "__main__":
    main()
