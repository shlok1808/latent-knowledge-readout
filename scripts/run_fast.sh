#!/usr/bin/env bash
# Parallel analysis (Shlok 2026-09-24 "make it fast"): uses the CPU cores left idle by tc2.
#   lane A: last-step Phase 3 (3L) on all depths -> then 3L on depth>=2 once its Phase 2/3 exist
#   lane B: depth>=2 Phase 2+3 now (tc2 would otherwise run it only after all-depths Phase 3;
#           probes.py's lock/DONE marker makes tc2 skip it when it gets there)
set -uo pipefail
cd ~/agent-failure-probes
R=runs/ms-tc-1p7b; O=results/multistep/textcraft_qwen3-1.7b; PY=.venv/bin/python
note() { echo "- $(date -u +%F\ %H:%M) UTC [fast] $*" | tee -a logs/runner_status.md; }
laneA() {
  note "start 3L all_depths"
  $PY -m mstep.probes --run $R --out $O/all_depths --phase 3L --jobs 8 >> logs/fast_3L.log 2>&1 && note "done 3L all_depths" || note "FAILED 3L all_depths"
  while [ ! -f $O/depth_ge2/DONE_all ]; do sleep 60; done
  note "start 3L depth_ge2"
  $PY -m mstep.probes --run $R --out $O/depth_ge2 --phase 3L --subset "depth>=2" --jobs 8 >> logs/fast_3L.log 2>&1 && note "done 3L depth_ge2" || note "FAILED 3L depth_ge2"
}
laneB() {
  note "start depth>=2 phase 2+3"
  $PY -m mstep.probes --run $R --out $O/depth_ge2 --subset "depth>=2" --jobs 10 >> logs/fast_d2.log 2>&1 && note "done depth>=2 phase 2+3" || note "FAILED depth>=2 phase 2+3"
}
laneA & A=$!; laneB & B=$!; wait $A $B
for d in all_depths depth_ge2; do [ -f $O/$d/phase3_transfer.csv ] && $PY -m mstep.figures --out $O/$d >> logs/fast_figs.log 2>&1; done
note "fast lanes complete (figures regenerated where Phase 3 exists)"
