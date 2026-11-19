"""Run bench.py as a notebook in the built site's Lab app, in each kernel (Chrome), and print its output.

    ./build.sh && uv run --group browser python spikes/rnn-epoch/run_in_lab.py

Uploads a one-cell notebook with the file browser (as the smoke test does for the Kaggle files), then opens it from the drive.
"""
import json, re, sys, tempfile, threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent.parent.parent
KERNEL_READY_JS = """() => {
  const status = document.querySelector('.jp-Notebook-ExecutionIndicator')?.dataset.status;
  if (status && status !== 'idle') window.__kernelStarting = true;
  return window.__kernelStarting === true && status === 'idle';
}"""

class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

nb = {"cells": [{"cell_type": "code", "execution_count": None, "id": "bench", "metadata": {}, "outputs": [],
                 "source": (Path(__file__).parent / "bench.py").read_text().splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python", "language": "python", "name": "python"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
tmp = Path(tempfile.mkdtemp()); (tmp / "bench.ipynb").write_text(json.dumps(nb))
server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=ROOT / "workspace" / "public"))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_port}"
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    for site in ("xeus", "pyodide"):
        page = browser.new_context().new_page()
        page.goto(f"{base}/{site}/lab/index.html")
        page.wait_for_selector("[title='Upload Files']", timeout=120000)
        with page.expect_file_chooser() as chooser:
            page.get_by_title("Upload Files").first.click()
        chooser.value.set_files(str(tmp / "bench.ipynb"))
        page.wait_for_selector("text=bench.ipynb", timeout=30000)
        page.goto(f"{base}/{site}/lab/index.html?path=bench.ipynb")
        page.wait_for_selector(".jp-CodeCell", timeout=240000)
        page.wait_for_function(KERNEL_READY_JS, timeout=240000, polling=100)
        page.click("li.lm-MenuBar-item:has-text('Run')"); page.click("li.lm-Menu-item:has-text('Run All Cells')")
        out = page.locator(".jp-OutputArea-output").first
        try:
            page.locator(".jp-CodeCell .jp-InputPrompt").last.filter(has_text=re.compile(r"\[\d+\]")).wait_for(timeout=900000)
            print(f"== {site}\n{out.inner_text()}")
        except Exception as e:
            print(f"== {site}: did not finish ({type(e).__name__})"); page.screenshot(path=f"/tmp/rnn-{site}.png")
    browser.close()
