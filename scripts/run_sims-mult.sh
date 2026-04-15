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

# --- Params ---
ITERATIONS=1
SYN_Threshold="0.1"

for ((i=1; i<=ITERATIONS; i++)); do
  echo "Running iteration $i"

  "$VENV_PY" "$SPIKES_FILE" N A
  "$VENV_PY" "$STDP_FILE" asym sym N 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "1.6 Comp PC new cue inh variance bci s $i" N 5000 "$SYN_Threshold" 1 A 1000
  "$VENV_PY" "$SPIKES_FILE" Y B
  "$VENV_PY" "$STDP_FILE" asym sym Y 2
  "$VENV_PY" "$PYTHON_FILE" asym sym "2nd env $i" N 5000 "$SYN_Threshold" 2 B 1000
  "$VENV_PY" "$PYTHON_FILE" asym sym "2x1st env $i" Y 5000 "$SYN_Threshold" 1 A 1000
  "$VENV_PY" "$PYTHON_FILE" asym sym "2x2nd env $i" Y 5000 "$SYN_Threshold" 2 B 1000
  # Sweep threshold (bash-safe float increment)
  #SYN_Threshold=$("$VENV_PY" -c 'import sys; print(f"{float(sys.argv[1]) + 0.25:.2f}")' "$SYN_Threshold")
done
