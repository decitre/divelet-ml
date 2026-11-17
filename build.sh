#!/usr/bin/env bash
# Build the divelet-ml JupyterLite sites into workspace/public/{xeus,pyodide}. Requires uv and micromamba.
# Adapted from jl-forge's examples/demo/build.sh. Each kernel builds in its own venv: jupyter lite
# copies every installed labextension, so a shared venv would put both kernels in both sites.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
WORKSPACE="$ROOT/workspace"
PUBLIC="$WORKSPACE/public"
# Kept between builds (CI caches it): xeus environments and Pyodide/self-hosted downloads.
CACHE="${JL_FORGE_CACHE:-$HOME/.cache/jl-forge}"
APPS=(--apps lab --apps notebooks --apps tree)
JL_FORGE="jl-forge==0.7.0"

venv() {  # venv <dir> <packages...>: create a build venv with jl-forge and jupyterlite
  local dir="$1"; shift
  uv venv -q "$dir"
  VIRTUAL_ENV="$dir" uv pip install -q "$JL_FORGE" jupyterlite-core jupyter_server jupyterlab_server libarchive-c "$@"
}

cd "$ROOT"
uv run --no-project --with "$JL_FORGE" jl-forge clean workspace

echo "== xeus-python site"
venv "$WORKSPACE/venv-xeus" "jupyterlite-xeus>=5,<6"
PATH="$WORKSPACE/venv-xeus/bin:$PATH"
cp -R content "$WORKSPACE/content-xeus"
jl-forge notebooks "$WORKSPACE/content-xeus" --strip
XEUS_PREFIX=$(jl-forge xeus-env --micromamba --cache-dir "$CACHE/xeus" --pyproject pyproject.toml)
# The divelet_ml package, mounted from source into the kernel's site-packages.
MOUNTS=$(jl-forge xeus-mount src/divelet_ml --prefix "$XEUS_PREFIX")
EMPACK_CONFIG=$(jl-forge xeus-pack-config --out "$WORKSPACE/empack.yml")
(cd "$WORKSPACE" && jupyter lite build "${APPS[@]}" --contents "$WORKSPACE/content-xeus" --output-dir "$PUBLIC/xeus" --lite-dir lite-xeus \
  --XeusAddon.prefix="$XEUS_PREFIX" \
  --XeusAddon.empack_config="$EMPACK_CONFIG" \
  --XeusAddon.mount_jupyterlite_content=True \
  --XeusAddon.mounts="$MOUNTS")

echo "== Pyodide site"
venv "$WORKSPACE/venv-pyodide" jupyterlite-pyodide-kernel==0.7.0
PATH="$WORKSPACE/venv-pyodide/bin:$PATH"
cp -R content "$WORKSPACE/content-pyodide"
# The divelet_ml package: one wheel and install.py at the contents root serve notebooks in every subdirectory.
uv build -q --wheel --out-dir "$WORKSPACE/content-pyodide" .
jl-forge pyodide-install --pyproject pyproject.toml --package-name divelet_ml --wheel-dir "$WORKSPACE/content-pyodide"
jl-forge notebooks "$WORKSPACE/content-pyodide" --strip --pyodide-install
jl-forge pyodide-wheels --pyproject pyproject.toml --cache-dir "$CACHE" --out-dir "$WORKSPACE/lite-pyodide/pypi"
# Self-host Pyodide, pruned to the kernel's startup packages, pyproject's dependencies and what the bundled wheels need.
PYODIDE_DIST=$(jl-forge pyodide-dist --pyproject pyproject.toml --wheels-dir "$WORKSPACE/lite-pyodide/pypi" --cache-dir "$CACHE/pyodide" --out "$WORKSPACE/pyodide-dist.tar.bz2")
(cd "$WORKSPACE" && jupyter lite build "${APPS[@]}" --contents "$WORKSPACE/content-pyodide" --output-dir "$PUBLIC/pyodide" --lite-dir lite-pyodide \
  --pyodide "$PYODIDE_DIST")

echo "== Fashion-MNIST"
# The dataset's four original files, served from the site's own host to the notebooks of 4.2 onwards (MIT license,
# Zalando SE). They stay out of content/, which xeus-python packs into the archive every notebook downloads.
FASHION_URL=https://github.com/zalandoresearch/fashion-mnist/raw/master
mkdir -p "$CACHE/fashion-mnist" "$PUBLIC/data/fashion-mnist"
while read -r sha name; do
  [ -f "$CACHE/fashion-mnist/$name" ] || curl -fsSL -o "$CACHE/fashion-mnist/$name" "$FASHION_URL/data/fashion/$name"
  echo "$sha  $CACHE/fashion-mnist/$name" | sha256sum -c --quiet -
  cp "$CACHE/fashion-mnist/$name" "$PUBLIC/data/fashion-mnist/"
done <<'SUMS'
3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84 train-images-idx3-ubyte.gz
a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845 train-labels-idx1-ubyte.gz
346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073 t10k-images-idx3-ubyte.gz
67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5 t10k-labels-idx1-ubyte.gz
SUMS
[ -f "$CACHE/fashion-mnist/LICENSE" ] || curl -fsSL -o "$CACHE/fashion-mnist/LICENSE" "$FASHION_URL/LICENSE"
cp "$CACHE/fashion-mnist/LICENSE" "$PUBLIC/data/fashion-mnist/LICENSE"

cat > "$PUBLIC/index.html" <<'HTML'
<!doctype html>
<meta charset="utf-8">
<title>divelet-ml</title>
<h1>divelet-ml — Dive into Deep Learning in JupyterLite</h1>
<p>A notebook link opens it with the xeus-python kernel; <b>[p]</b> opens it with Pyodide instead.</p>
<ul>
HTML
(cd content && find . -name '*.ipynb' | sed 's|^\./||' | sort) | while read -r nb; do
  echo "  <li><a href=\"xeus/notebooks/index.html?path=$nb\"><code>$nb</code></a> <a href=\"pyodide/notebooks/index.html?path=$nb\">[p]</a></li>"
done >> "$PUBLIC/index.html"
cat >> "$PUBLIC/index.html" <<'HTML'
</ul>
<footer>
<p>Adapted from <a href="https://d2l.ai/"><i>Dive into Deep Learning</i></a> by Aston Zhang, Zachary C. Lipton, Mu Li,
Alexander J. Smola and contributors. Adapted book text:
<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>; each notebook lists its source and changes.
divelet-ml code: <a href="https://github.com/decitre/divelet-ml/blob/main/LICENSE">Apache License 2.0</a>.</p>
<p>Not affiliated with or endorsed by the book's authors, or by D2L Corporation.
Source: <a href="https://github.com/decitre/divelet-ml">github.com/decitre/divelet-ml</a>.</p>
</footer>
HTML
touch "$PUBLIC/.nojekyll"

jl-forge self-host "$PUBLIC" --cache-dir "$CACHE/self-host"
jl-forge dedup "$PUBLIC"
jl-forge size "$PUBLIC" --max 1G
jl-forge urls "$PUBLIC"
