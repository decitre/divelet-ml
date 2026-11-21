"""Every notebook in content/ runs in CPython without errors, warnings or other stderr output.

    uv run --group test pytest tests/test_notebooks.py

A fast check before the browser smoke test; it does not show that a notebook runs in JupyterLite.
"""

import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = sorted((Path(__file__).parent.parent / "content").rglob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)  # a cell type such as "md" is not rejected by `read`, and Lab shows that cell as raw text
    NotebookClient(nb, timeout=300, kernel_name="python3").execute()
    stderr = [o["text"] for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.get("name") == "stderr"]
    assert not stderr


# Book markup that a conversion can leave behind: the highlight markers `[**...**]` and `(**...**)` and the d2lbook directives show as
# literal text in JupyterLab.
HIGHLIGHT = [re.compile(r"\[\*\*.+?\*\*\](?!\()", re.S), re.compile(r"\(\*\*.+?\*\*\)", re.S)]
DIRECTIVE = re.compile(r":(numref|eqref|label|cite|citet|begin_tab|end_tab|width|height|eqlabel):|\.\./img/|\[\[CITE")
CODE_LEFTOVER = re.compile(r"tab\.selected|%%tab|^from d2l import|^import torch|^from torch", re.M)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_has_no_book_markup_left(path):
    nb = nbformat.read(path, as_version=4)
    prose = [re.sub(r"```.*?```", "", c.source, flags=re.S) for c in nb.cells if c.cell_type == "markdown"]  # code shown in a note is left alone
    left = [m.group(0)[:70] for text in prose for rx in HIGHLIGHT + [DIRECTIVE] for m in rx.finditer(text)]
    left += [m.group(0) for c in nb.cells if c.cell_type == "code" for m in CODE_LEFTOVER.finditer(c.source)]
    assert not left, left
