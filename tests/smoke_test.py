"""Browser smoke test: every notebook in content/ runs in each kernel and app, with no error output
and nothing fetched from other hosts.

Each case runs in desktop Google Chrome and in WebKit (Safari's engine, which every iPad browser uses)
emulating an iPad in portrait. The emulation covers the engine, screen size and touch, not a real
tablet's memory limits.

Run after build.sh, from the repository root (needs Google Chrome; install WebKit once with
`uv run --group browser playwright install webkit`):

    uv run --group browser pytest tests/smoke_test.py -v

Adapted from jl-forge's examples/demo/smoke_test.py.
"""

import hashlib
import json
import re
import threading
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent.parent
PUBLIC = ROOT / "workspace" / "public"
FAILURES = ROOT / "workspace" / "smoke-failures"  # screenshots of notebooks that did not finish
FOLDER_LISTING = re.compile(r"/api/contents/.+/all\.json$")
NOTEBOOKS = sorted(p.relative_to(ROOT / "content").as_posix() for p in (ROOT / "content").rglob("*.ipynb"))
KERNEL_START_MS = 240_000  # first load downloads the kernel and its packages
RUN_MS = 120_000
# The two notebooks that train two multilayer perceptrons take 1.5 to 3 minutes in Pyodide on a CI runner, in WebKit; the dropout one has gone past 5.
RUN_MS_NOTEBOOKS = {"05_multilayer-perceptrons/05.02_mlp-implementation.ipynb": 300_000,
                    "05_multilayer-perceptrons/05.06_dropout.ipynb": 480_000,
                    # Training the recurrent language models for the browser's share of the book's 100 epochs: about a minute in Chrome on one Mac.
                    "09_recurrent-neural-networks/09.05_rnn-scratch.ipynb": 300_000,
                    "09_recurrent-neural-networks/09.06_rnn-concise.ipynb": 300_000}
# The kernel name each site's toolbar shows: catches a site built with the other kernel.
KERNEL_NAMES = {"xeus": "(XPython)", "pyodide": "(Pyodide)"}
IPAD = "iPad (gen 11)"
# The toolbar's execution indicator reads "idle" before any kernel exists, then "initializing" or "unknown"
# while the kernel starts, then "idle" again once it is ready: wait for that second "idle".
KERNEL_READY_JS = """() => {
  const status = document.querySelector('.jp-Notebook-ExecutionIndicator')?.dataset.status;
  if (status && status !== 'idle') window.__kernelStarting = true;
  return window.__kernelStarting === true && status === 'idle';
}"""
# Notebooks that need files the site cannot host: the reader downloads them from the book's host and uploads them with
# the file browser, so the test does the same (in the Lab app; the Notebook app has no file browser).
UPLOADS = {"05_multilayer-perceptrons/05.07_kaggle-house-price.ipynb": {
    "kaggle_house_pred_train.csv": "585e9cc93e70b39160e7921475f9bcd7d31219ce",
    "kaggle_house_pred_test.csv": "fa19780a7b011d9b009e8bff8e99922a8ee2eb90"}}
CASES = [(b, site, app, nb) for b in ("chrome", "ipad") for site in ("xeus", "pyodide")
         for app in ("lab", "notebooks") for nb in NOTEBOOKS]


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def base_url():
    if not (PUBLIC / "index.html").is_file():
        pytest.skip(f"{PUBLIC} not built; run ./build.sh first")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=PUBLIC))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(scope="session")
def upload_files(tmp_path_factory):
    """upload_files(notebook): the paths of the files the notebook needs, downloaded and checked."""
    def upload_files(notebook):
        folder = tmp_path_factory.mktemp("uploads")
        paths = []
        for name, sha1 in UPLOADS[notebook].items():
            path = folder / name
            if not path.exists():
                path.write_bytes(urllib.request.urlopen(f"https://d2l-data.s3.amazonaws.com/{name}").read())
            assert hashlib.sha1(path.read_bytes()).hexdigest() == sha1
            paths.append(str(path))
        return paths
    return upload_files


@pytest.fixture(scope="module")
def new_context():
    """new_context(name): a fresh browser context, launching each browser on first use."""
    with sync_playwright() as p:
        browsers = {}

        def new_context(name):
            if name not in browsers:
                browsers[name] = p.webkit.launch() if name == "ipad" else p.chromium.launch(channel="chrome")
            return browsers[name].new_context(**(p.devices[IPAD] if name == "ipad" else {}))

        yield new_context
        for browser in browsers.values():
            browser.close()


@pytest.mark.parametrize(("browser", "site", "app", "notebook"), CASES,
                         ids=[f"{b}-{s}-{a}-{Path(n).stem}" for b, s, a, n in CASES])
def test_notebook(new_context, upload_files, base_url, browser, site, app, notebook):
    if notebook in UPLOADS and app != "lab":
        pytest.skip("needs files uploaded with the file browser, which only the Lab app has")
    context = new_context(browser)
    page = context.new_page()
    responses, console_errors, page_errors = [], [], []
    page.on("response", lambda r: responses.append((r.url, r.status)))
    page.on("pageerror", lambda e: page_errors.append(str(e)))
    # The kernel's own fetches run in a web worker; their failures show up here, not always as responses.
    # The browser's own /favicon.ico request at the server root is outside the site.
    # Listings of folders a notebook creates (2.2's ../data): Pyodide's drive asks the site for them.
    page.on("console", lambda m: console_errors.append(m.text)
            if m.type == "error" and not m.location.get("url", "").endswith("/favicon.ico")
            and not FOLDER_LISTING.search(m.location.get("url", "")) else None)
    page.goto(f"{base_url}/{site}/{app}/index.html?path={notebook}")
    # A notebook without code (3.6) still opens with a kernel; there is just nothing to run.
    has_code = any(c["cell_type"] == "code" for c in json.loads((ROOT / "content" / notebook).read_text())["cells"])
    page.wait_for_selector(".jp-CodeCell" if has_code else ".jp-MarkdownCell", timeout=KERNEL_START_MS)
    # Run All while the kernel is still starting can lose the session in WebKit: nothing runs, or Lab
    # shows a "Select Kernel" dialog. A tablet user who taps Run All right away sees the same.
    page.wait_for_function(KERNEL_READY_JS, timeout=KERNEL_START_MS, polling=100)
    errors = page.locator(".jp-OutputArea-output[data-mime-type='application/vnd.jupyter.stderr']")
    if notebook in UPLOADS:
        with page.expect_file_chooser() as chooser:
            page.get_by_title("Upload Files").first.click()
        chooser.value.set_files(upload_files(notebook))
        # Lab opens the uploaded CSV files in tabs over the notebook: come back to it.
        page.locator(".lm-DockPanel-tabBar .lm-TabBar-tab", has_text=Path(notebook).stem.split("_")[0]).first.click()
    if has_code:
        _run_all(page, errors, browser, site, app, notebook, page_errors, console_errors)
    assert errors.count() == 0, errors.all_inner_texts()
    assert KERNEL_NAMES[site] in page.locator(".jp-Toolbar-kernelName").first.inner_text()

    # Self-hosted: no other host, and no file missing from the site (how a pruned Pyodide package shows up).
    assert {urlparse(u).netloc for u, _ in responses if not u.startswith(base_url)} == set()
    assert not [(u, s) for u, s in responses if u.startswith(base_url) and s >= 400 and not FOLDER_LISTING.search(u)]
    assert not console_errors
    context.close()


def _run_all(page, errors, browser, site, app, notebook, page_errors, console_errors):
    page.click("li.lm-MenuBar-item:has-text('Run')")
    page.click("li.lm-Menu-item:has-text('Run All Cells')")

    # Run All stops at the first error, so the last code cell showing "[n]:" (not "[ ]:" or "[*]:")
    # means every cell ran. Off-screen prompts can render empty, so don't check them all.
    # Stop waiting at the first error output instead of timing out.
    last_ran = page.locator(".jp-CodeCell .jp-InputPrompt").last.filter(has_text=re.compile(r"\[\d+\]"))
    try:
        last_ran.or_(errors.first).first.wait_for(timeout=RUN_MS_NOTEBOOKS.get(notebook, RUN_MS))
    except PlaywrightTimeoutError:
        FAILURES.mkdir(parents=True, exist_ok=True)
        shot = FAILURES / f"{browser}-{site}-{app}-{Path(notebook).stem}.png"
        page.screenshot(path=shot)
        prompts = page.locator(".jp-CodeCell .jp-InputPrompt").all_inner_texts()
        status = page.locator(".jp-Notebook-ExecutionIndicator").first.get_attribute("data-status")
        pytest.fail(f"cells did not finish: prompts {prompts}, kernel {status}, page errors {page_errors}, "
                    f"console errors {console_errors}, screenshot {shot}")
