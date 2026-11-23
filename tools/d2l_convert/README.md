# The converter

The tool that drafts a notebook from a section of [*Dive into Deep Learning*](https://d2l.ai/) and builds the notebooks of a chapter. It is what every notebook in `content/` started from;
the notebooks are then adapted by hand (see the skill [`adapt-d2l-section`](../../.claude/skills/adapt-d2l-section/SKILL.md) for the conventions and the checks).

Nothing here is needed to read, run or build the site. It is for adapting more of the book.

## What it does

`convert.draft(path)` reads a section's Markdown and returns `(title, cells, forum)`, where `cells` is a list of `("md" | "code", source)`:

- keeps the code of the PyTorch tab only: a cell that names its tabs on its first line (`%%tab pytorch, mxnet` or `#@tab all`) is kept when the list has `pytorch` or `all`, and a cell with no marker is shared;
  the tab machinery (`%load_ext d2lbook.tab`, `tab.interact_select`) and `:begin_tab:` blocks for other frameworks are dropped, and the PyTorch tab's `[Discussions](...)` link becomes `forum`;
- turns `:numref:` and `:eqref:` into links with the numbers the book shows (the labels of the whole book are indexed by `d2lindex.py`), numbers every display equation (`\tag{10.1.4}`), and drops `:label:`, `:width:` and `:height:` lines;
- writes `:cite:` and `:citet:` as "(Author et al., 2014)" and "Author et al. (2014)" from `d2l.bib`;
- strips the highlight markers, `[**text**]` and `(**text**)` (a bold link, `[**text**](url)`, stays);
- embeds the book's images as data URIs, except the two it reproduces from papers (`THIRD_PARTY`), which become a link;
- collects what it could not resolve (an unknown label or citation) in `convert.WARN`.

It does not adapt code: that is the part that needs a person (`import torch` and `d2l.torch` become NumPy, autograd and `divelet_ml`; see the skill).

`nbbuild.Chapter.build(...)` then writes the notebook of a section: the header credit cell, the navigation cells, the section's cells with your adapted code cells in place of the draft's,
notes that the book does not have (marked with the tablet and the cell tag `divelet-ml`), the forum cell, and the "Attribution and changes" cell with the list of changes.
`example_bi_rnn.py` rebuilds Section 10.4 with it, and `tests/test_converter.py` checks that the result equals the notebook in `content/` (the notebooks are edited by hand after they are built, so a rebuilt one is a starting point).

## Use

The book's sources are pinned at commit `23d7a5aecceee57d1292c56e90cce307f183bb0a` of `d2l-ai/d2l-en` (`convert.C`):

```bash
git clone https://github.com/d2l-ai/d2l-en.git
git -C d2l-en checkout 23d7a5aecceee57d1292c56e90cce307f183bb0a

# The draft of a section, with its cell numbers (the keys of the `code` argument of `Chapter.build`)
D2L_SRC=d2l-en python tools/d2l_convert/convert.py chapter_recurrent-modern/lstm

# The label index (how many labels and sections it found)
D2L_SRC=d2l-en python tools/d2l_convert/d2lindex.py
```

`D2L_SRC` is the checkout (default `./d2l-en`); `D2L_IMG_CACHE` is where the images are kept (default `~/.cache/divelet-ml/d2l-img`; they are downloaded from the pinned commit). Only the standard library is needed.

## What it finds in the whole book

Run over every section listed in the book's tables of contents (163 of the 165 section files: `chapter_hyperparameter-optimization/rs-async.md` is in none, and `chapter_references/zreferences.md`
is the bibliography), it drafts all of them without an error, and keeps 1,190 code cells of the PyTorch tab or shared by every tab. Five sections give warnings, eight in all, all in chapters that are not adapted yet:

| Section | Warning |
|---|---|
| `chapter_natural-language-processing-applications/natural-language-inference-attention.md` | unknown label `eq_nli_v_ab` |
| `chapter_gaussian-processes/gp-priors.md` | unknown label `eq_gp-function` |
| `chapter_hyperparameter-optimization/hyperopt-intro.md` | unknown citation `hutter-book19a`; unknown label `sec_rs_async` |
| `chapter_hyperparameter-optimization/sh-intro.md`, `sh-async.md` | unknown label `sec_rs_async` |

The adapted chapters (2 to 10, 11.3 and 12) give none. The warnings are for whoever adapts Chapters 16, 18 and 19 to look at: the labels are either in the book's own text under another form, or dangling in the book.
