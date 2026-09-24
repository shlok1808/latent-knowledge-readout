#!/usr/bin/env bash
# Stage-2 replication check (Shlok 2026-09-24): depth-1 episodes only, action-token site
# (stage-2 / Ruan's probe position), Phase 2 only, full layer sweep, so the
# round-3 / layer-12 cell is directly comparable to stage-2's 0.801 within-task AUC.
# Stage-2's text baseline included round t's observation; ours excludes it (A-15).
set -uo pipefail
cd ~/agent-failure-probes
OUT=results/multistep/textcraft_qwen3-1.7b/depth_eq1_action
note() { echo "- $(date -u +%F\ %H:%M) UTC [depth1-repl] $*" | tee -a logs/runner_status.md; }
note "start: depth==1, site=action, phase 2"
if .venv/bin/python -m mstep.probes --run runs/ms-tc-1p7b --out $OUT --phase 2 --site action --subset "depth==1" --jobs ${JOBS:-8} >> logs/depth1_replication.log 2>&1; then
  note "done: $(.venv/bin/python -c "
import pandas as pd
c=pd.read_csv('$OUT/phase2_cells.csv'); b=pd.read_csv('$OUT/phase2_baselines.csv')
x=c[(c.label=='final_bad')&(c.input=='S1')&(c.probe=='logreg')&(c.t==3)&(c.layer==12)].iloc[0]
tb=b[(b.label=='final_bad')&(b.t==3)&(b.baseline=='tfidf_surface')].iloc[0]
print(f'L12 t3 within={x.within_auc:.3f} [{x.within_lo:.2f},{x.within_hi:.2f}] pooled={x.pooled_auc:.3f} mixed={int(x.mixed_tasks)}; text t3 within={tb.within_auc:.3f}; stage-2 was 0.801 vs text 0.725')")"
else
  note "FAILED (see logs/depth1_replication.log)"
fi
