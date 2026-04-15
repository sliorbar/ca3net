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
SYN_Threshold="2.50"

for ((i=1; i<=ITERATIONS; i++)); do
  echo "Running iteration $i"

  "$VENV_PY" "$SPIKES_FILE" N A
  "$VENV_PY" "$STDP_FILE" asym sym N 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "PC PC-contx plast replay 2 env 1 $i" N 5000 "$SYN_Threshold" 1 A 20000 Y
  "$VENV_PY" "$SPIKES_FILE" Y B
  "$VENV_PY" "$STDP_FILE" asym sym Y 2
  "$VENV_PY" "$PYTHON_FILE" asym sym "PC PC-contx plast replay 2 env 2 $i" N 5000 "$SYN_Threshold" 2 B 20000 Y
  "$VENV_PY" "$PYTHON_FILE" asym sym "2.5 PC PC-contx plast replay A $i" Y 5000 "$SYN_Threshold" 1 A 1000 N
  "$VENV_PY" "$PYTHON_FILE" asym sym "2.5 PC PC-contx plast replay B $i" Y 5000 "$SYN_Threshold" 2 B 1000 N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "2.0 PC PC-inh-contx plast only replay $i" Y 5000 "$SYN_Threshold" 1 A 1000 N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "2.25 PC PC-inh-contx plast only replay $i" Y 5000 "$SYN_Threshold" 1 A 1000 N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "2.5 PC PC-inh-contx plast only replay $i" Y 5000 "$SYN_Threshold" 1 A 1000 N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "2.75 PC plast only replay $i" Y 5000 "$SYN_Threshold" 1 A 1000 N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "3.0 PC plast only replay $i" Y 5000 3.0 1 A 1000 N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "3.25 PC plast only replay $i" Y 5000 3.25 1 A 1000 N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "3.5 PC plast only replay $i" Y 5000 3.5 1 A 1000 N

  
  # Sweep threshold (bash-safe float increment)
  #SYN_Threshold=$("$VENV_PY" -c 'import sys; print(f"{float(sys.argv[1]) + 0.2:.2f}")' "$SYN_Threshold")
done
