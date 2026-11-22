"""Run step_cost.py in both JupyterLite kernels of a built divelet-ml site (workspace/public), in Chrome, and print its output."""
import json, re, sys, threading, time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
PUB = HERE.parent.parent / "workspace" / "public"
INSTALL = 'import sys; sys.path.append("/drive"); from install import _; await _()\n'
READY = """() => {const s=document.querySelector('.jp-Notebook-ExecutionIndicator')?.dataset.status;
  if (s && s !== 'idle') window.__k = true; return window.__k === true && s === 'idle';}"""

class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

def notebook(site):
    src = (HERE / "step_cost.py").read_text()
    code = (INSTALL if site == "pyodide" else "") + src
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": code.splitlines(keepends=True), "outputs": [], "execution_count": None}],
          "metadata": {"kernelspec": {"display_name": "Python", "language": "python", "name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = HERE / f"_step_cost_{site}.ipynb"; path.write_text(json.dumps(nb)); return path

srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=PUB)); threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}"
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    for site in sys.argv[1:] or ["xeus", "pyodide"]:
        path = notebook(site)
        page = b.new_context().new_page()
        page.goto(f"{base}/{site}/lab/index.html"); page.wait_for_selector(".jp-FileBrowser", timeout=240000)
        with page.expect_file_chooser() as fc: page.get_by_title("Upload Files").first.click()
        fc.value.set_files(str(path)); page.wait_for_timeout(3000)
        page.goto(f"{base}/{site}/lab/index.html?path={path.name}")
        page.wait_for_selector(".jp-CodeCell", timeout=240000)
        page.locator(".jp-Dialog button:has-text('Select')").click(timeout=30000)  # an uploaded notebook asks which kernel
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
