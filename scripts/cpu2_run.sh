#!/usr/bin/env bash
# Runs ON the CPU helper box (GH200, 64 cores). Minimal venv, then the depth>=2 analyses:
#   phase 3 (range-averaged + milestone + per-round curves, layers from depth_ge2 Phase 2) and
#   phase 3X (layers 12,13,15,16,28 x all three schemes), in parallel. Writes ~/cpu2_done at the end.
set -uo pipefail
cd ~/agent-failure-probes
exec >> ~/cpu2_run.log 2>&1
echo "== start $(date -u)"
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ~/venv ] || { uv venv -q ~/venv --python 3.12 && VIRTUAL_ENV=~/venv uv pip install -q numpy pandas scikit-learn scipy joblib matplotlib threadpoolctl; }
PY=~/venv/bin/python; R=runs/ms-tc-1p7b; O=results/multistep/textcraft_qwen3-1.7b/depth_ge2
$PY -c "import numpy, pandas, sklearn; print('deps ok', numpy.__version__, sklearn.__version__)"
( $PY -m mstep.probes --run $R --out $O --phase 3 --subset "depth>=2" --jobs 30 > ~/cpu2_p3.log 2>&1; echo "phase3 exit=$?" ) &
( $PY -m mstep.probes --run $R --out $O --phase 3X --subset "depth>=2" --layers 12,13,15,16,28 --jobs 30 > ~/cpu2_p3x.log 2>&1; echo "phase3X exit=$?" ) &
wait
echo "== done $(date -u)"
touch ~/cpu2_done
