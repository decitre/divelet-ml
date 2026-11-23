"""Draft the cells of a notebook from a section of Dive into Deep Learning: the PyTorch tabs kept, d2lbook directives turned into links,
citations written out, highlight markers stripped, images embedded. See README.md.

    D2L_SRC=d2l-en python tools/d2l_convert/convert.py chapter_recurrent-modern/lstm
"""
import re, json, base64, uuid, urllib.request, sys, os, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2lindex import LABELS, FILES, SRC
C = "23d7a5aecceee57d1292c56e90cce307f183bb0a"  # the commit of d2l-ai/d2l-en that the sections are taken from
IMG_CACHE = os.environ.get("D2L_IMG_CACHE", os.path.expanduser("~/.cache/divelet-ml/d2l-img"))
WARN = []


def _balanced(text, start):
    """The text of the braces or quotes that open at `start`."""
    if text[start] == '"':
        return text[start + 1:text.index('"', start + 1)]
    depth, i = 0, start
    while True:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start + 1:i]
        i += 1


def _clean(name):
    accents = {'"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303", "c": "\u0327", "v": "\u030c"}
    name = re.sub(r"\\([\"'`^~cv])\s*\{?(\w)\}?", lambda m: unicodedata.normalize("NFC", m.group(2) + accents[m.group(1)]), name)
    return re.sub(r"[{}\\]", "", name).strip()


def load_bib():
    bib, text = {}, open(SRC + "d2l.bib").read()
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+),", text):
        end = text.find("\n@", m.end())
        body = text[m.end():end if end > 0 else len(text)]
        a, y = re.search(r"\bauthor\s*=\s*", body, re.I), re.search(r"\byear\s*=\s*[{\"]?\s*(\d{4})", body, re.I)
        if a and y:
            bib[m.group(1)] = (_balanced(body, a.end()), y.group(1))
    return bib


BIB = load_bib()


def author_year(key):
    entry = BIB.get(re.sub(r"\*\d+$", "", key))
    if not entry:
        WARN.append(f"unknown citation {key}")
        return key, ""
    names = [n.strip() for n in re.split(r"\s+and\s+", re.sub(r"[\s~]+", " ", entry[0]))]
    others = bool(names) and names[-1] == "others"
    last = [_clean(n.split(",")[0] if "," in n else n.split()[-1]) for n in names if n != "others"]
    who = f"{last[0]} et al." if others or len(last) > 2 else " and ".join(last)
    return who, entry[1]


def cite(keys, narrative):
    pairs = [author_year(k.strip()) for k in keys.split(",")]
    if narrative:
        return " and ".join(f"{who} ({year})" for who, year in pairs)
    return "(" + "; ".join(f"{who}, {year}" for who, year in pairs) + ")"


# Figures that the book reproduces from papers (its captions credit the authors): not embedded, only linked.
THIRD_PARTY = {"filters.png", "regnet-fig.png"}


def ref(kind, lab):
    k = LABELS.get(lab)
    if not k:
        WARN.append(f"unknown label {lab}")
        return f"[{lab}]"
    if kind == "numref":
        return f"[{ {'fig': 'Fig. ', 'chap': 'Chapter '}.get(k[0], 'Section ')}{k[1]}]({k[2]})"
    return f"[({k[1]})]({k[2]})"


def embed(path):
    cache = os.path.join(IMG_CACHE, os.path.basename(path))
    os.makedirs(IMG_CACHE, exist_ok=True)
    if not os.path.exists(cache):
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/d2l-ai/d2l-en/{C}/img/{os.path.basename(path)}", cache)
    ext = cache.rsplit(".", 1)[-1].lower()
    mime = {"svg": "image/svg+xml", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[ext]
    return f"data:{mime};base64," + base64.b64encode(open(cache, "rb").read()).decode()


def tag_equations(text, base):
    k = [0]

    def sub(m):
        k[0] += 1
        return f"$${m.group(1).rstrip()} \\tag{{{base}.{k[0]}}}$$"
    return re.sub(r"\$\$(.*?)\$\$(?:\n:eqlabel:`[^`]+`)?", sub, text, flags=re.S)


def clean_md(t):
    t = re.sub(r"^:(label|width|height):`?[^\n]*\n?", "", t, flags=re.M)
    t = re.sub(r":(numref|eqref):`([^`]+)`", lambda m: ref(m.group(1), m.group(2)), t)
    def figure(m):
        if m.group(2) in THIRD_PARTY:
            return f"*Figure not reproduced here (third-party image): {m.group(1)} See [the book's figure](https://d2l.ai/_images/{m.group(2)}).*"
        return f"![{m.group(1)}]({embed(m.group(2))})"
    t = re.sub(r"!\[([^\]]*)\]\(\.\./img/([^)]+)\)", figure, t)
    t = re.sub(r":cite:`([^`]+)`", lambda m: cite(m.group(1), False), t)
    t = re.sub(r":citet:`([^`]+)`", lambda m: cite(m.group(1), True), t)
    # The book's highlight markers, [**text**] and (**text**), become text (a bold link, [**text**](url), stays).
    t = re.sub(r"\[\*\*((?:(?!\*\*\]).)+)\*\*\](?!\()", r"\1", t, flags=re.S)  # the text cannot hold `**]`, so a bold link does not start a span
    t = re.sub(r"\(\*\*(.+?)\*\*\)", r"\1", t, flags=re.S)
    return t.strip("\n")


def draft(path):
    ci, si = FILES[path]
    base = f"{ci}.{si}"
    text = tag_equations(open(SRC + path).read(), base)
    lines = text.splitlines()
    segs, md, i, forum, title = [], [], 0, None, None

    def flush():
        if md and "\n".join(md).strip():
            segs.append(("md", "\n".join(md)))
        md.clear()
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            block = lines[i + 1:j]
            if line.startswith("```{.python"):
                tabs = None
                m_tab = re.match(r"\s*#?\s*(?:%%tab|@tab)\s+(.*)", block[0]) if block else None  # `%%tab pytorch`, `# %%tab jax` or `#@tab all`
                if m_tab:
                    tabs = {x.strip() for x in m_tab.group(1).split(",")}
                    block = block[1:]
                keep = (tabs is None) or ("pytorch" in tabs) or ("all" in tabs)
                if block and block[0].startswith(("%load_ext", "tab.interact_select")):
                    keep = False
                if keep and "".join(block).strip():
                    flush()
                    segs.append(("code", "\n".join(block)))
            elif not line.startswith("```eval_rst") and not line.startswith("```toc"):
                md += lines[i:j + 1]
            i = j + 1
            continue
        m = re.match(r":begin_tab:`([^`]+)`", line)
        if m:
            names = {x.strip() for x in m.group(1).split(",")}
            j = i + 1
            while not lines[j].startswith(":end_tab:"):
                j += 1
            body = "\n".join(lines[i + 1:j])
            if "pytorch" in names:
                d = re.search(r"\[Discussions\]\(https://discuss\.d2l\.ai/t/(\d+)\)", body)
                if d:
                    forum = d.group(1)
                else:
                    md += lines[i + 1:j]
            i = j + 1
            continue
        if title is None and line.startswith("# "):
            title = line[2:].strip()
        md.append(line)
        i += 1
    flush()
    return title, [(k, clean_md(v) if k == "md" else v) for k, v in segs], forum


def cell(kind, src):
    c = {"cell_type": {"md": "markdown"}.get(kind, kind), "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


if __name__ == "__main__":
    path = sys.argv[1] if sys.argv[1].endswith(".md") else sys.argv[1] + ".md"  # e.g. chapter_recurrent-modern/lstm
    title, cells, forum = draft(path)
    print("TITLE:", title, "| forum:", forum, "| warnings:", WARN)
    for n, (k, v) in enumerate(cells):
        print(f"--- [{n}] {k}")
        print(v if k == "code" else v[:300].replace("\n", "\n    ") + (" ..." if len(v) > 300 else ""))
