#!/usr/bin/env bash
# Full DW4 experiment: 2 architectures x 3 couplings, run in parallel on CPU, then figures.
# Usage:  nohup bash run_all.sh > results/run_all.log 2>&1 &
# Quick check (a few minutes):  QUICK=1 bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
mkdir -p results/logs

if [[ "${QUICK:-0}" == "1" ]]; then
    ARGS="--epochs 2 --n-train 4096 --n-val 1024 --n-eval 1000"
    PAIRS_ARGS="--n-train 4096 --n-val 1024 --n-sets 2"
else
    ARGS=""
    PAIRS_ARGS=""
fi

$PY scripts/prepare_dw4.py
echo "[$(date +%T)] precomputing equivariant-OT pairs"
$PY scripts/precompute_eq_ot.py $PAIRS_ARGS

echo "[$(date +%T)] training 6 configurations in parallel"
for arch in egnn mlp; do
    for coupling in independent ot eq_ot; do
        OMP_NUM_THREADS=3 $PY scripts/run_experiment.py --arch $arch --coupling $coupling --threads 3 $ARGS \
            > results/logs/${arch}_${coupling}.log 2>&1 &
    done
done
wait
echo "[$(date +%T)] all runs finished"

$PY scripts/make_figures.py
echo "[$(date +%T)] done: see results/summary.md, results/fig_distributions.png, results/fig_nfe.png"
