"""The notebooks that train less in the browser each have a figure of the full run after their training cell.

The figures come from tools/full_run_figures.py, which runs the notebooks; this only checks that they are in place, so that a new
notebook that trains less in the browser, or an edit that drops a figure, does not go unnoticed.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("full_run_figures", ROOT / "tools" / "full_run_figures.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def cells(rel):
    return json.loads((ROOT / "content" / rel).read_text())["cells"]


def test_every_notebook_that_trains_less_in_the_browser_is_a_target():
    users = {p.relative_to(ROOT / "content").as_posix() for p in (ROOT / "content").rglob("*.ipynb")
             if any(c["cell_type"] == "code" and ("browser_epochs(" in "".join(c["source"]) or "browser_examples(" in "".join(c["source"]))
                    for c in cells(p.relative_to(ROOT / "content")))}
    assert users == set(tool.TARGETS)


def test_each_training_cell_is_followed_by_its_full_run_figure():
    for rel in tool.TARGETS:
        cs = cells(rel)
        training = [i for i, c in enumerate(cs) if c["cell_type"] == "code" and "trainer.fit(" in "".join(c["source"])]
        assert training, rel
        for i in training:
            figure = cs[i + 1]
            text = "".join(figure["source"])
            assert figure["cell_type"] == "markdown" and {"divelet-ml", "full-run"} <= set(figure["metadata"]["tags"]), (rel, i)
            assert "**Full run, for comparison.**" in text and "](data:image/svg+xml;base64," in text, (rel, i)
        assert sum("full-run" in c.get("metadata", {}).get("tags", []) for c in cs) == len(training), rel
