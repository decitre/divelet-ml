"""Index the book's labels at the pinned commit: label -> (kind, number, url), and FILES: section path -> (chapter number, section number).

`D2L_SRC` is a checkout of https://github.com/d2l-ai/d2l-en at the commit `convert.C` (default: `./d2l-en`); see README.md."""
import re, os
SRC = os.environ.get("D2L_SRC", "d2l-en").rstrip("/") + "/"
def toc(path):
    t = open(SRC + path).read()
    out = []
    for block in re.findall(r"```toc\n(.*?)```", t, re.S):
        out += [l.strip() for l in block.splitlines() if l.strip() and not l.startswith(":")]
    return out
chapters = [c for c in toc("index.md") if c.endswith("/index") and c.startswith("chapter_")]
chapters = chapters[chapters.index("chapter_introduction/index"):]  # numbered chapters start at the introduction
LABELS, FILES = {}, {}
def url(chapdir, name, label=None):
    return f"https://d2l.ai/{chapdir}/{name}.html" + (f"#{label.replace('_', '-')}" if label else "")
for ci, chap in enumerate(chapters, 1):
    chapdir = chap.split("/")[0]
    for si, entry in enumerate(toc(f"{chapdir}/index.md"), 1):
        name = entry
        path = f"{chapdir}/{name}.md"
        if not os.path.exists(SRC + path): continue
        FILES[path] = (ci, si)
        base = f"{ci}.{si}"
        h = [0, 0]; fig = 0; eq = 0; lines = open(SRC + path).read().splitlines()
        last_heading = base; in_code = False
        for i, line in enumerate(lines):
            if line.startswith("```"): in_code = not in_code
            if in_code: continue
            m = re.match(r"(#{1,3}) ", line)
            if m:
                lvl = len(m.group(1))
                if lvl == 1: last_heading = base
                elif lvl == 2: h[0] += 1; h[1] = 0; last_heading = f"{base}.{h[0]}"
                else: h[1] += 1; last_heading = f"{base}.{h[0]}.{h[1]}"
            for lab in re.findall(r":label:`([^`]+)`", line):
                if lab.startswith("fig_"):
                    fig += 1; LABELS[lab] = ("fig", f"{base}.{fig}", url(chapdir, name, lab))
                elif lab.startswith("tab_"):
                    LABELS[lab] = ("tab", None, url(chapdir, name, lab))
                else:
                    LABELS[lab] = ("sec", last_heading, url(chapdir, name, lab))
# Chapter labels live in each chapter's index.md.
for ci, chap in enumerate(chapters, 1):
    chapdir = chap.split("/")[0]
    for lab in re.findall(r":label:`([^`]+)`", open(SRC + f"{chapdir}/index.md").read()):
        LABELS[lab] = ("chap", str(ci), f"https://d2l.ai/{chapdir}/index.html#{lab.replace('_', '-')}")
# Every display equation `$$...$$` of a section is numbered, labelled or not.
for path, (ci, si) in FILES.items():
    chapdir, name = path[:-3].split("/")
    text = open(SRC + path).read()
    for k, m in enumerate(re.finditer(r"\$\$(.*?)\$\$(\n:eqlabel:`([^`]+)`)?", text, re.S), 1):
        if m.group(3):
            LABELS[m.group(3)] = ("eq", f"{ci}.{si}.{k}", url(chapdir, name, "equation-" + m.group(3)))
if __name__ == "__main__":
    print(len(LABELS), "labels,", len(FILES), "sections")
