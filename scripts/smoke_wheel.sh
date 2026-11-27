#!/usr/bin/env bash
# Install a built wheel into a fresh environment and run it away from the checkout, the way a user
# would after `uv tool install opentrain-cli`.
#
#   scripts/smoke_wheel.sh dist/opentrain_cli-*.whl
set -euo pipefail

wheel="$(realpath "$1")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv venv --quiet "$work/venv"
uv pip install --quiet --python "$work/venv/bin/python" "$wheel"

cd "$work"
export HOME="$work/home" XDG_STATE_HOME="$work/state" XDG_DATA_HOME="$work/data" XDG_CONFIG_HOME="$work/config"
mkdir -p "$HOME"

"$work/venv/bin/python" - <<'PY'
import importlib.metadata
from pathlib import Path

import opentrain
from opentrain.core.skills import bundled_dir

assert opentrain.__version__ == importlib.metadata.version("opentrain-cli"), opentrain.__version__
web = Path(opentrain.__file__).parent / "web" / "app" / "index.html"
assert web.is_file(), "the web app is missing from the wheel"
skills = {p.parent.name for p in bundled_dir().glob("*/SKILL.md")}
for name in ("goal-coding", "data-diagnose", "data-history", "data-find", "data-tasks", "data-triage"):
    assert name in skills, f"skill {name} is missing from the wheel"
from opentrain.traces.readers import scan
assert all(r["count"] == 0 for r in scan()), "a clean home has no agent logs"
print(f"wheel smoke test passed: opentrain {opentrain.__version__}, {len(skills)} skills, web app present")
PY

"$work/venv/bin/opentrain" --version
"$work/venv/bin/opentrain-cli" traces | grep -q "No coding-agent logs found"
echo "wheel smoke test passed: opentrain traces"
