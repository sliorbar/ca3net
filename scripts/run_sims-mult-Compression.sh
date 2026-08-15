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
ITERATIONS=1
SYN_Threshold="2.5" # Initial synaptic threshold for compression (will be incremented in each iteration)

for ((i=1; i<=ITERATIONS; i++)); do
  echo "Running iteration $i"

  "$VENV_PY" "$SPIKES_FILE" N A
  "$VENV_PY" "$STDP_FILE" asym sym N 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold inh plast 300 bcs new min A env - Org Order $i" N 5000 "$SYN_Threshold" 1 A 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R B
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold inh plast 300 bcs new min B env random order $i" N 5000 "$SYN_Threshold" 2 B 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R C
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold inh plast 300 bcs new min C env random order $i" N 5000 "$SYN_Threshold" 2 C 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R D
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold 300 bcs new min D env random order $i" N 5000 "$SYN_Threshold" 2 D 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R E
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold 300 bcs new min E env random order $i" N 5000 "$SYN_Threshold" 2 E 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R F
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold 300 bcs new min F env random order $i" N 5000 "$SYN_Threshold" 2 F 20000 Y N N
  "$VENV_PY" "$SPIKES_FILE" R G
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold 300 bcs new min G env random order $i" N 5000 "$SYN_Threshold" 2 G 20000 Y N N
  "$VENV_PY" "$SPIKES_FILE" R H
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold 300 bcs new min H env random order $i" N 5000 "$SYN_Threshold" 2 H 20000 Y N N
  "$VENV_PY" "$SPIKES_FILE" R I
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold 300 bcs new min I env random order $i" N 5000 "$SYN_Threshold" 2 I 20000 Y N N
  
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold No PC plast E env random order $i" N 5000 "$SYN_Threshold" 2 E 5000 Y N

  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold inh plast 300 bcs new min preprocess mult env A $i" Y 5000 "$SYN_Threshold" 1 A 1000 N N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold inh plast 300 bcs new min preprocess mult env B $i" Y 5000 "$SYN_Threshold" 1 B 1000 N N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold inh plast 300 bcs new min preprocess mult env I $i" Y 5000 "$SYN_Threshold" 1 I 1000 N N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC plast preprocess mult env D $i" Y 5000 "$SYN_Threshold" 1 D 1000 N N Y
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC plast preprocess mult env E $i" Y 5000 "$SYN_Threshold" 1 E 1000 N N Y
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC plast preprocess mult env no cue E $i" N 5000 "$SYN_Threshold" 1 E 10000 N N Y
  #"$VENV_PY" "$PYTHON_FILE" asym sym "1.75 Hom no PC plast only replay $i" Y 5000 1.75 1 A 1000 N N Y
  #"$VENV_PY" "$PYTHON_FILE" asym sym "2.0 Hom no PC plast only replay $i" Y 5000 2.0 1 A 1000 N N Y
  #"$VENV_PY" "$PYTHON_FILE" asym sym "2.25 Hom no PC plast only replay $i" Y 5000 2.25 1 A 1000 N N Y
  #"$VENV_PY" "$PYTHON_FILE" asym sym "2.5 Hom no PC plast only replay $i" Y 5000 2.5 1 A 1000 N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "2.75 Hom no PC plast only replay $i" Y 5000 2.75 1 A 1000 N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "3.0 Hom no PC plast only replay $i" Y 5000 3.0 1 A 1000 N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "3.25 Hom no PC plast only replay $i" Y 5000 3.25 1 A 1000 N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "3.5 Hom no PC plast only replay $i" Y 5000 3.5 1 A 1000 N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "3.75 Hom no PC plast only replay $i" Y 5000 3.75 1 A 1000 N N

  
  # Sweep threshold (bash-safe float increment)
  # SYN_Threshold=$("$VENV_PY" -c 'import sys; print(f"{float(sys.argv[1]) + 0.25:.2f}")' "$SYN_Threshold")
done
