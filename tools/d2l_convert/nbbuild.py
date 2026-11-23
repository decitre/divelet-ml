"""Build the notebooks of one chapter from `convert.draft` and the hand-written NumPy code of each section. See README.md.

A chapter is described once; each section is then built with the code cells that need adapting, the list of changes that its
"Attribution and changes" cell prints, and the notes that the book does not have (marked with the tablet and the cell tag `divelet-ml`).

    chapter = Chapter(number=10, book_dir="chapter_recurrent-modern", out_dir="content/10_recurrent-modern", sections=[...], adapted={1, 2, 3, 4},
                      previous="Previous: [Section 9.7: ...](../09_.../09.07_bptt.ipynb)")
    chapter.build("lstm", code={1: "import numpy as np"}, changes=[NUMPY, LIB])
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert  # noqa: E402
from convert import C as BOOK_COMMIT, cell  # noqa: E402

TAG = '<img src="../tablet.svg" alt="divelet-ml" title="Added by divelet-ml, not in the book"> '
LIB = ("`d2l` is divelet-ml's package (`import divelet_ml as d2l`) instead of the book's `d2l.torch`. "
       "`%matplotlib inline` is removed: both JupyterLite kernels display plots inline.")
NUMPY = "Code uses NumPy instead of PyTorch, so it runs in JupyterLite (Pyodide and xeus-python)."
COMMON = ["Equations keep the book's numbers. Cross-references to other sections link to the book's site, and citations are written inline (author and year); the book's bibliography lists them.",
          "The d2lbook framework tabs for MXNet, TensorFlow and JAX are removed."]


class Chapter:
    def __init__(self, number, book_dir, out_dir, sections, adapted, previous=None, next_=None):
        """`sections` is a list of (file name without `.md`, title) in the book's order; `adapted` the numbers of the ones that exist, so that
        the navigation cells link to those only; `previous` and `next_` are the navigation text of the first and last section to the
        neighbouring chapters (a Markdown link), or None."""
        self.number, self.book_dir, self.out_dir, self.sections, self.adapted = number, book_dir, out_dir, sections, set(adapted)
        self.previous, self.next = previous, next_
        self.files = {name: f"{number}.{i:02d}_{name}.ipynb" for i, (name, _) in enumerate(sections, 1)}

    def nav(self, name):
        i = [n for n, _ in self.sections].index(name) + 1
        parts = []
        if i == 1 and self.previous:
            parts.append(self.previous)
        elif i - 1 in self.adapted:
            parts.append(f"Previous: [Section {self.number}.{i - 1}: {self.sections[i - 2][1]}]({self.files[self.sections[i - 2][0]]})")
        if i == len(self.sections) and self.next:
            parts.append(self.next)
        elif i + 1 in self.adapted:
            parts.append(f"Next: [Section {self.number}.{i + 1}: {self.sections[i][1]}]({self.files[self.sections[i][0]]})")
        return " | ".join(parts)

    def build(self, name, code, changes, figures=False, common=None, added=()):
        """Write the notebook of section `name`.

        `code`: {index of a code cell of the draft: its adapted source}; every cell listed must exist (a draft's cell indices are printed by
        `python convert.py <section>`), and the cells not listed are kept as they are.
        `changes`: the bullets of "Changes from the original" (plain Markdown strings), then `common` (default: `COMMON`).
        `added`: (index of a draft cell, text) or (index, text, code): a note before that cell, with the tablet and the tag `divelet-ml`, and a code
        cell after it if `code` is given.
        `figures`: say "Text and figures" in the license line (it is set when the draft has an embedded image)."""
        title = dict(self.sections)[name]
        path = f"{self.book_dir}/{name}.md"
        convert.WARN.clear()
        _, cells, forum = convert.draft(path)
        assert not convert.WARN, convert.WARN
        if forum is None:  # a plain "[Discussions](...)" line at the end, outside the framework tabs
            for n, (k, v) in enumerate(cells):
                m = re.search(r"\n*\[Discussions\]\(https://discuss\.d2l\.ai/t/(\d+)\)\s*$", v)
                if k == "md" and m:
                    forum = m.group(1)
                    cells[n] = (k, v[:m.start()].rstrip())
            assert forum, "no forum link"
        figures = figures or any("data:image" in v for k, v in cells)
        out = [cell("markdown", f"> Adapted from *Dive into Deep Learning*, section [{title}](https://d2l.ai/{self.book_dir}/{name}.html). Not affiliated with or endorsed by the book's authors.\n"
                                "> Source, authors, licenses and changes: [Attribution and changes](#Attribution-and-changes), at the end of this notebook."),
               cell("markdown", self.nav(name))]
        used = set()
        for i, (k, v) in enumerate(cells):
            for a in added:
                if a[0] == i:
                    note = cell("markdown", TAG + a[1])
                    note["metadata"] = {"tags": ["divelet-ml"]}
                    out.append(note)
                    if len(a) > 2:
                        out.append(cell("code", a[2]))
            if k == "code" and i in code:
                used.add(i)
                v = code[i]
            out.append(cell(k, v))
        assert used == set(code), f"unused replacements {set(code) - used}; code cells are {[i for i, (k, _) in enumerate(cells) if k == 'code']}"
        out.append(cell("markdown", f"Discussions of the original section: [book's forum](https://discuss.d2l.ai/t/{forum})"))
        out.append(cell("markdown", self.nav(name)))
        kind = "Text and figures" if figures else "Text"
        out.append(cell("markdown", f"""## Attribution and changes

Adapted from *Dive into Deep Learning*, section
[{title}](https://d2l.ai/{self.book_dir}/{name}.html)
(source: [`{path}`](https://github.com/d2l-ai/d2l-en/blob/{BOOK_COMMIT}/{path})
at commit `{BOOK_COMMIT[:12]}`), by Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola and contributors.
{kind}: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Code: adapted from the book's sample code, MIT-0.
Not affiliated with or endorsed by the book's authors.

**Changes from the original** (PyTorch edition):
""" + "\n".join("- " + c for c in changes + (COMMON if common is None else common))))
        nb = {"cells": out, "metadata": {"kernelspec": {"display_name": "Python", "language": "python", "name": "python"}, "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        os.makedirs(self.out_dir, exist_ok=True)
        target = os.path.join(self.out_dir, self.files[name])
        with open(target, "w") as f:
            f.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        return target
