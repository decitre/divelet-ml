"""Add to the notebooks that train less in the browser a figure of what the book's full training gives.

    uv run --group test python tools/full_run_figures.py [NOTEBOOK ...]

For each notebook below, this runs it in CPython with the book's settings (where `d2l.browser_epochs` and `d2l.browser_examples`
return everything), takes the progress plot that each training cell ends with, and writes it, with the final values of
its curves, into a markdown cell right after that cell. The cell carries the tags `divelet-ml` and `full-run`, so running
the script again replaces it. The random start is seeded, so the same code gives the same figure; run the script again
after changing the library or a training cell, and commit the notebooks it changes.
"""
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SEED = 0
TAG = '<img src="../tablet.svg" alt="divelet-ml" title="Added by divelet-ml, not in the book"> '
# notebook -> (the book's settings, for the caption; source replacements in the training cells that give them)
TARGETS = {
    "05_multilayer-perceptrons/05.02_mlp-implementation.ipynb": ("10 epochs, as in the book", []),
    "05_multilayer-perceptrons/05.06_dropout.ipynb": ("10 epochs, as in the book", []),
    "07_convolutional-neural-networks/07.06_lenet.ipynb": (
        "10 epochs on all 60,000 training images, as in the book",
        [("num_train=d2l.browser_examples(10000, pyodide=2000, xeus=10000)", "num_train=None")]),
    "08_convolutional-modern/08.05_batch-norm.ipynb": (
        "10 epochs on all 60,000 training images, as in the book",
        [("num_train=d2l.browser_examples(10000, pyodide=2000, xeus=10000)", "num_train=None")]),
    "12_optimization/12.11_lr-scheduler.ipynb": (
        "30 epochs on all 60,000 training images and 10,000 test images, as in the book",
        [("d2l.browser_examples(2000, pyodide=1000, xeus=2000)", "None")]),
    "09_recurrent-neural-networks/09.05_rnn-scratch.ipynb": ("100 epochs, as in the book", []),
    "09_recurrent-neural-networks/09.06_rnn-concise.ipynb": ("100 epochs, as in the book", []),
    "10_recurrent-modern/10.01_lstm.ipynb": ("50 epochs, as in the book", []),
    "10_recurrent-modern/10.02_gru.ipynb": ("50 epochs, as in the book", []),
    "10_recurrent-modern/10.03_deep-rnn.ipynb": ("100 epochs, as in the book", []),
}


# Draws the model's board once, into a new figure: the board redraws all of its lines at every update, so the figure that the
# training cell ends with holds hundreds of overlapping lines and is several times larger.
DRAW = """import io
import matplotlib.pyplot as plt
board = model.board
fig, ax = plt.subplots(figsize=board.figsize)
for (label, line), ls, color in zip(board.data.items(), board.ls, board.colors):
    ax.plot([p.x for p in line], [p.y for p in line], linestyle=ls, color=color, label=label)
if board.xlim: ax.set_xlim(board.xlim)
if board.ylim: ax.set_ylim(board.ylim)
ax.set_xlabel(board.xlabel); ax.set_ylabel(board.ylabel); ax.set_xscale(board.xscale); ax.set_yscale(board.yscale); ax.legend()
svg = io.StringIO(); fig.savefig(svg, format='svg', bbox_inches='tight'); plt.close(fig)
print(json.dumps({'svg': svg.getvalue(), 'values': {k: float(v[-1].y) for k, v in board.data.items()}}))"""


def is_training(cell):
    """A cell that trains: `trainer.fit(` (the framework's `Trainer`, with a board) or a call of a notebook's own `train(net, ...)`
    function (with an `Animator`, as in 12.11)."""
    source = cell["source"] if isinstance(cell, dict) else cell.source
    source = "".join(source)
    kind = cell["cell_type"] if isinstance(cell, dict) else cell.cell_type
    return kind == "code" and ("trainer.fit(" in source or re.search(r"^train\(net,", source, re.M) is not None)


# The Animator's figure, which the cell that trained still holds when it ends: kept in `_fig` by a line added to the cell.
KEEP = "\nimport matplotlib.pyplot as _plt\n_fig = _plt.gcf()"
DRAW_ANIMATOR = """import io
svg = io.StringIO(); _fig.savefig(svg, format='svg', bbox_inches='tight')
print(json.dumps({'svg': svg.getvalue()}))"""
FINAL = re.compile(r"train loss ([\d.]+), train acc ([\d.]+), test acc ([\d.]+)")


def run(path, replacements):
    """Execute a copy of the notebook with the book's settings, in its own folder. A cell after each training cell draws the model's
    board and prints the figure with the final point of every curve. Returns [(svg of the last figure, {curve: final value})], one per
    training cell, in order."""
    nb = nbformat.read(path, as_version=4)
    cells, trained = [nbformat.v4.new_code_cell(
        f"import json, matplotlib, numpy\nnumpy.random.seed({SEED})\n"
        "# Text as text, not as paths, and fixed ids: a smaller figure that the same run reproduces exactly\n"
        "matplotlib.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'divelet-ml'})")], []
    for cell in nb.cells:
        if cell.cell_type == "markdown" and "full-run" in cell.metadata.get("tags", []):
            continue  # the figures of an earlier run
        cells.append(cell)
        if cell.cell_type == "code":  # the data can be loaded in an earlier cell than the training
            for old, new in replacements:
                cell.source = cell.source.replace(old, new)
        if is_training(cell):
            animator = "trainer.fit(" not in cell.source
            if animator:
                cell.source += KEEP
            trained.append(len(cells) - 1)
            cells.append(nbformat.v4.new_code_cell(DRAW_ANIMATOR if animator else DRAW))
    nb.cells = cells
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    found = []
    for i in trained:
        out = json.loads("".join(o["text"] for o in nb.cells[i + 1].outputs if o.get("name") == "stdout"))
        if "values" not in out:  # the Animator's figure: the final values are in the line the training function prints
            printed = "".join(o["text"] for o in nb.cells[i].outputs if o.get("name") == "stdout")
            out["values"] = dict(zip(["train loss", "train acc", "test acc"], map(float, FINAL.search(printed).groups())))
        found.append((re.sub(r"<metadata>.*?</metadata>\s*", "", out["svg"], flags=re.S), out["values"]))  # the metadata holds the date
    return found


def figure_cell(rel, n, settings, svg, values):
    final = ", ".join(f"{label} {value:.3f}" for label, value in values.items())
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    text = (TAG + "**Full run, for comparison.** In JupyterLite, the cell above trains less than the book does (see the note before it). "
            f"This figure is what it gives with {settings}: computed with CPython, not in the browser, from a random start (seed {SEED}), "
            f"so a run of your own will differ in the details. Final values: {final}.\n\n"
            f"![Training curves of the full run. Final values: {final}.]({uri})")
    return {"cell_type": "markdown", "id": hashlib.sha1(f"{rel}:{n}".encode()).hexdigest()[:8],
            "metadata": {"tags": ["divelet-ml", "full-run"]}, "source": text.splitlines(keepends=True)}


def update(rel):
    path = ROOT / "content" / rel
    settings, replacements = TARGETS[rel]
    found = run(path, replacements)
    nb = json.loads(path.read_text())
    cells = [c for c in nb["cells"] if "full-run" not in c.get("metadata", {}).get("tags", [])]
    positions = [i for i, c in enumerate(cells) if is_training(c)]
    assert len(positions) == len(found), (rel, positions, len(found))
    for n, (i, (svg, values)) in reversed(list(enumerate(zip(positions, found)))):
        cells.insert(i + 1, figure_cell(rel, n, settings, svg, values))
    nb["cells"] = cells
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"{rel}: {len(found)} figure(s); final values {[v for _, v in found]}")


if __name__ == "__main__":
    for rel in sys.argv[1:] or TARGETS:
        update(rel)
