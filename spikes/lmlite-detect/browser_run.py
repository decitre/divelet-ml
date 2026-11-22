"""Upload detect.ipynb to lmlite's hosted JupyterLite site, run all its cells in Chrome, print the outputs."""
import re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
BASE = "https://decitre.github.io/lmlite"
READY = """() => {const s=document.querySelector('.jp-Notebook-ExecutionIndicator')?.dataset.status;
  if (s && s !== 'idle') window.__k = true; return window.__k === true && s === 'idle';}"""
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    for site in sys.argv[1:] or ["xeus", "pyodide"]:
        ctx = b.new_context(); page = ctx.new_page()
        errs = []
        page.on("console", lambda m: errs.append(m.text[:200]) if m.type == "error" else None)
        t0 = time.time()
        page.goto(f"{BASE}/{site}/lab/index.html")
        page.wait_for_selector(".jp-FileBrowser", timeout=240000)
        with page.expect_file_chooser() as fc:
            page.get_by_title("Upload Files").first.click()
        nbname = "detect.ipynb" if site == "xeus" else "detect_pyodide.ipynb"
        fc.value.set_files(str(Path(__file__).parent / nbname))
        page.wait_for_timeout(4000)
        page.goto(f"{BASE}/{site}/lab/index.html?path={nbname}")
        page.wait_for_selector(".jp-CodeCell", timeout=240000)
        page.wait_for_function(READY, timeout=240000, polling=100)
        t1 = time.time()
        page.click("li.lm-MenuBar-item:has-text('Run')"); page.click("li.lm-Menu-item:has-text('Run All Cells')")
        last = page.locator(".jp-CodeCell .jp-InputPrompt").last.filter(has_text=re.compile(r"\[\d+\]"))
        err = page.locator(".jp-OutputArea-output[data-mime-type='application/vnd.jupyter.stderr']")
        last.or_(err.first).first.wait_for(timeout=300000)
        page.wait_for_timeout(1500)
        print(f"===== {site}: kernel ready after {t1 - t0:.0f} s, all cells ran in {time.time() - t1:.0f} s")
        for o in page.locator(".jp-OutputArea-output").all_inner_texts(): print(o.strip())
        print("stderr:", err.all_inner_texts()[:2], "console errors:", errs[:3])
        page.screenshot(path=str(Path(__file__).parent / f"browser-{site}.png"))
        ctx.close()
    b.close()
