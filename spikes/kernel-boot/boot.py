"""Spike: what does it cost to open a notebook in the built site, cold and with the browser's cache warm?

    ./build.sh && uv run --group browser python spikes/kernel-boot/boot.py

For each kernel, in Chrome: open a notebook of the Notebook app as the README links do, wait for the kernel to be ready, and
run its first code cell (where the Pyodide notebooks install `divelet_ml` and its dependencies), and count the bytes the server
sent. Then open a second notebook in the same browser context (warm cache), the first one again, and finally the second one
in a fresh context (cold). The server sends `Cache-Control: max-age=600`, which is what I believe GitHub Pages sends, and
counts the bytes it sends per file: the kernels run in a web worker, whose requests Playwright's page events do not report,
and a response served from the browser's cache never reaches the server.
"""
import collections
import tempfile
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

PUBLIC = Path(__file__).resolve().parent.parent.parent / "workspace" / "public"
A, B = "02_preliminaries/02.01_ndarray.ipynb", "02_preliminaries/02.02_data-preprocessing.ipynb"  # B imports pandas
KERNEL_READY_JS = """() => {
  const status = document.querySelector('.jp-Notebook-ExecutionIndicator')?.dataset.status;
  if (status && status !== 'idle') window.__kernelStarting = true;
  return window.__kernelStarting === true && status === 'idle';
}"""


SENT = []  # (path, bytes) of every response the server sent


class Handler(SimpleHTTPRequestHandler):
    def send_header(self, keyword, value):
        if keyword.lower() == "content-length":
            SENT.append((self.path.split("?")[0], int(value)))
        super().send_header(keyword, value)

    def end_headers(self):
        self.send_header("Cache-Control", "max-age=600")
        super().end_headers()

    def log_message(self, *args):
        pass


def kind(path):
    return "wasm" if path.endswith(".wasm") else "packages" if path.endswith((".whl", ".tar.gz", ".zip")) else "js/css/other"


def boot(context, base, site, notebook):
    SENT.clear()
    page = context.new_page()
    t = time.time()
    page.goto(f"{base}/{site}/notebooks/index.html?path={notebook}")
    page.wait_for_selector(".jp-MarkdownCell", timeout=240000)
    page.wait_for_function(KERNEL_READY_JS, timeout=240000, polling=100)
    ready = time.time() - t
    page.locator(".jp-CodeCell").first.click()
    page.keyboard.press("Shift+Enter")  # the first code cell: the Pyodide notebooks install divelet_ml and its dependencies here
    page.locator(".jp-CodeCell .jp-InputPrompt").first.filter(has_text="[1]").wait_for(timeout=240000)
    first_cell = time.time() - t
    page.wait_for_timeout(500)
    page.close()
    kinds = collections.Counter()
    for path, size in SENT:
        kinds[kind(path)] += size
    return ready, first_cell, sum(s for _, s in SENT), len(SENT), kinds


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=PUBLIC))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    with sync_playwright() as p:
        # A persistent profile has a disk cache; the throwaway contexts of `browser.new_context()` keep their cache in memory,
        # where the largest files (the Pyodide and xeus wasm, packages over a few MB) do not fit and are fetched again.
        profile = lambda: p.chromium.launch_persistent_context(tempfile.mkdtemp(), channel="chrome")
        for site in ("pyodide", "xeus"):
            warm = profile()
            print(f"== {site}")
            for label, context, notebook in (("1st notebook, cold", warm, A), ("2nd notebook, warm cache", warm, B),
                                             ("1st notebook again, warm", warm, A), ("2nd notebook, cold (fresh profile)", profile(), B)):
                ready, first_cell, total, n, kinds = boot(context, base, site, notebook)
                print(f"  {label:36s} kernel ready {ready:5.1f} s, first cell done {first_cell:5.1f} s; {total / 1e6:6.1f} MB sent in {n} requests "
                      f"({', '.join(f'{k} {v / 1e6:.1f} MB' for k, v in kinds.items())})", flush=True)


if __name__ == "__main__":
    main()
