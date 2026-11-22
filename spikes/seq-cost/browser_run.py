"""Run step_cost.py or text_cost.py (the first argument) in both JupyterLite kernels of a built divelet-ml site (workspace/public), in Chrome, and print its output."""
import json, re, sys, threading, time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
PUB = HERE.parent.parent / "workspace" / "public"
INSTALL = 'import sys; sys.path.append("/drive"); from install import _; await _()\n'
# The kernel is ready when the toolbar names it (not "No Kernel") and its execution indicator is idle. The dialog wait above can outlast
# the kernel's start, so the smoke test's "saw it change, then idle" check would miss it.
READY = """() => {const s=document.querySelector('.jp-Notebook-ExecutionIndicator')?.dataset.status;
  const k=document.querySelector('.jp-Toolbar-kernelName')?.textContent||''; return s === 'idle' && /Pyodide|XPython/.test(k);}"""

class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

def notebook(site, script):
    src = (HERE / script).read_text()
    code = (INSTALL if site == "pyodide" else "") + src
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": code.splitlines(keepends=True), "outputs": [], "execution_count": None}],
          "metadata": {"kernelspec": {"display_name": "Python", "language": "python", "name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = HERE / f"_{Path(script).stem}_{site}.ipynb"; path.write_text(json.dumps(nb)); return path

srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=PUB)); threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}"
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    script, sites = sys.argv[1], sys.argv[2:] or ["xeus", "pyodide"]  # e.g. step_cost.py xeus pyodide
    for site in sites:
        path = notebook(site, script)
        page = b.new_context().new_page()
        page.goto(f"{base}/{site}/lab/index.html"); page.wait_for_selector(".jp-FileBrowser", timeout=240000)
        with page.expect_file_chooser() as fc: page.get_by_title("Upload Files").first.click()
        fc.value.set_files(str(path)); page.wait_for_timeout(3000)
        page.goto(f"{base}/{site}/lab/index.html?path={path.name}")
        page.wait_for_selector(".jp-CodeCell", timeout=240000)
        try:  # an uploaded notebook asks which kernel in the xeus-python site; the Pyodide site has only one
            page.locator(".jp-Dialog button:has-text('Select')").click(timeout=15000)
        except Exception:
            pass
        try:
            page.wait_for_function(READY, timeout=240000, polling=100)
        except Exception:
            page.screenshot(path=str(HERE / f"debug-{site}.png")); raise
        t = time.time()
        page.click("li.lm-MenuBar-item:has-text('Run')"); page.click("li.lm-Menu-item:has-text('Run All Cells')")
        page.wait_for_timeout(5000)
        page.wait_for_function("document.querySelector('.jp-Notebook-ExecutionIndicator')?.dataset.status==='idle'", timeout=3_000_000, polling=1000)
        print(f"===== {site}: {time.time() - t:.0f} s")
        print("\n".join(o.strip() for o in page.locator(".jp-OutputArea-output").all_inner_texts()))
        path.unlink()
    b.close()
