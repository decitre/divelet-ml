"""The converter of tools/d2l_convert, on a small made-up book (and, when D2L_SRC points at the real sources, on Section 10.4).

The converter reads the book from the folder `D2L_SRC` at import time, so each check runs it in a subprocess.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
TOOL = ROOT / "tools" / "d2l_convert"

BIB = """@article{smith2020,
  author = {Smith, Ann and Jones, Bob and Lee, Cy},
  year = {2020}
}
@inproceedings{pair2019,
  author = "Ng, Di and Park, Ed",
  year = 2019
}
"""
SECTION = """# A Section
:label:`sec_x`

See :numref:`sec_x`, :numref:`fig_a` and :eqref:`eq_one`, and also :numref:`sec_missing`.
:cite:`smith2020,pair2019` and :citet:`pair2019`.

The [**highlighted**] and (**marked**) words, a [**bold link**](https://example.com), and
[**a span
over lines**] too.

![A picture.](../img/a.svg)
:label:`fig_a`
:width:`300px`

$$x = 1$$
:eqlabel:`eq_one`

$$y = 2$$

```{.python .input}
%%tab all
shared = 1
```

```{.python .input}
%%tab pytorch, mxnet
both = 2
```

```{.python .input}
#@tab mxnet
only_mxnet = 3
```

```{.python .input}
#@tab all
hash_all = 4
```

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch')
```

```{.python .input}
no_marker = 5
```

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:
"""
SNIPPET = """
import json, sys
sys.path.insert(0, %r)
import convert
title, cells, forum = convert.draft('chapter_demo/sec.md')
print(json.dumps({'title': title, 'cells': cells, 'forum': forum, 'warnings': convert.WARN}))
"""


@pytest.fixture(scope="module")
def draft(tmp_path_factory):
    book = tmp_path_factory.mktemp("book")
    (book / "index.md").write_text("```toc\nchapter_introduction/index\nchapter_demo/index\n```\n")
    (book / "chapter_introduction").mkdir()
    (book / "chapter_introduction" / "index.md").write_text("# Introduction\n```toc\n```\n")
    (book / "chapter_demo").mkdir()
    (book / "chapter_demo" / "index.md").write_text("# Demo\n```toc\nsec\n```\n")
    (book / "chapter_demo" / "sec.md").write_text(SECTION)
    (book / "d2l.bib").write_text(BIB)
    img = book / "img"
    img.mkdir()
    (img / "a.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    env = {**os.environ, "D2L_SRC": str(book), "D2L_IMG_CACHE": str(img)}
    out = subprocess.run([sys.executable, "-c", SNIPPET % str(TOOL)], env=env, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_only_the_pytorch_and_shared_code_cells_are_kept(draft):
    code = [v for k, v in draft["cells"] if k == "code"]
    assert code == ["shared = 1", "both = 2", "hash_all = 4", "no_marker = 5"]  # not the MXNet-only cell, nor the tab machinery; the marker lines are gone
    assert draft["forum"] == "42" and draft["title"] == "A Section"


def test_references_equations_and_citations_become_links_and_text(draft):
    text = "\n".join(v for k, v in draft["cells"] if k == "md")
    assert "[Section 2.1](https://d2l.ai/chapter_demo/sec.html#sec-x)" in text  # the introduction is chapter 1, the demo chapter 2
    assert "[Fig. 2.1.1](https://d2l.ai/chapter_demo/sec.html#fig-a)" in text
    assert "[(2.1.1)](https://d2l.ai/chapter_demo/sec.html#equation-eq-one)" in text
    assert "$$x = 1 \\tag{2.1.1}$$" in text and "$$y = 2 \\tag{2.1.2}$$" in text  # every display equation is numbered, labelled or not
    assert "(Smith et al., 2020; Ng and Park, 2019)" in text and "Ng and Park (2019)" in text
    assert ":numref:" not in text and ":cite" not in text and ":label:" not in text and ":width:" not in text
    assert "![A picture.](data:image/svg+xml;base64," in text


def test_highlight_markers_are_stripped_but_bold_links_stay(draft):
    text = "\n".join(v for k, v in draft["cells"] if k == "md")
    assert "The highlighted and marked words" in text and "[**bold link**](https://example.com)" in text
    assert "a span\nover lines too" in text and "[**" not in text.replace("[**bold link**]", "") and "(**" not in text


def test_an_unknown_label_is_a_warning_not_an_error(draft):
    assert draft["warnings"] == ["unknown label sec_missing"]


@pytest.mark.skipif(not (Path(os.environ.get("D2L_SRC", "d2l-en")) / "chapter_recurrent-modern" / "bi-rnn.md").exists(),
                    reason="needs the book's sources at the pinned commit: set D2L_SRC (see tools/d2l_convert/README.md)")
def test_section_10_4_is_rebuilt_exactly(tmp_path):
    env = {**os.environ, "D2L_IMG_CACHE": str(tmp_path / "img")}
    subprocess.run([sys.executable, str(TOOL / "example_bi_rnn.py"), str(tmp_path)], env=env, check=True, capture_output=True)
    strip = lambda nb: [{k: v for k, v in c.items() if k != "id"} for c in nb["cells"]]  # cell ids are random
    built = json.loads((tmp_path / "10.04_bi-rnn.ipynb").read_text())
    kept = json.loads((ROOT / "content" / "10_recurrent-modern" / "10.04_bi-rnn.ipynb").read_text())
    assert strip(built) == strip(kept) and built["metadata"] == kept["metadata"]
