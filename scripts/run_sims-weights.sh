#!/usr/bin/env bash
set -euo pipefail

# --- Paths (WSL) ---
REPO="$HOME/projects/ca3net"
VENV_PY="$REPO/.venv/bin/python"

#PYTHON_FILE="$REPO/scripts/replay_stats_simulations-param-sweep.py"
#PYTHON_FILE="$REPO/scripts/replay_stats_simulations-inhibitory-plasticity.py"
PYTHON_FILE="$REPO/scripts/replay_stats_simulations-inhibitory-plasticity-mult.py"
STDP_FILE="$REPO/scripts/stdp-Inh-mult.py"
#STDP_FILE="$REPO/scripts/stdp-Inh.py"
#STDP_FILE="$REPO/scripts/stdp.py"
SPIKES_FILE="$REPO/scripts/generate_spike_train-Omen.py"
#SPIKES_FILE="$REPO/scripts/generate_spike_train-random.py"

# --- Params ---
ITERATIONS=10
SYN_Threshold="0.1" # Initial synaptic threshold for compression (will be incremented in each iteration)

for ((i=1; i<=ITERATIONS; i++)); do
  echo "Running iteration $i"

  "$VENV_PY" "$SPIKES_FILE" N A
  "$VENV_PY" "$STDP_FILE" asym sym N 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact no plast $i" N 5000 "$SYN_Threshold" 1 A 5000 Y N Y
  "$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 6s $i" N 5000 "$SYN_Threshold" 1 A 1000 Y N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 10s $i" N 5000 "$SYN_Threshold" 1 A 5000 Y N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 15s $i" N 5000 "$SYN_Threshold" 1 A 10000 Y N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 20s $i" N 5000 "$SYN_Threshold" 1 A 15000 Y N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 25s $i" N 5000 "$SYN_Threshold" 1 A 20000 Y N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 30s $i" N 5000 "$SYN_Threshold" 1 A 25000 Y N N

    # Sweep threshold (bash-safe float increment)
  #SYN_Threshold=$("$VENV_PY" -c 'import sys; print(f"{float(sys.argv[1]) + 0.25:.2f}")' "$SYN_Threshold")
done
