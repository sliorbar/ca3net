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
SYN_Threshold="2.0" # Initial synaptic threshold for compression (will be incremented in each iteration)

for ((i=1; i<=ITERATIONS; i++)); do
  echo "Running iteration $i"

  "$VENV_PY" "$SPIKES_FILE" N A "" 0.35
  "$VENV_PY" "$STDP_FILE" asym sym N 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax A env - Org Order $i" N 5000 "$SYN_Threshold" 1 A 20000 Y N N
  "$VENV_PY" "$SPIKES_FILE" R B "" 0.35
  "$VENV_PY" "$STDP_FILE" asym sym Y 2
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax B env - Org Order $i" N 5000 "$SYN_Threshold" 2 B 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R C "" 0.35
  "$VENV_PY" "$STDP_FILE" asym sym Y 2
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax C env - Org Order $i" N 5000 "$SYN_Threshold" 2 C 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R D "" 0.35
  "$VENV_PY" "$STDP_FILE" asym sym Y 2
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax D env - Org Order $i" N 5000 "$SYN_Threshold" 2 D 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R E "" 0.35
  "$VENV_PY" "$STDP_FILE" asym sym Y 2
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax E env - Org Order $i" N 5000 "$SYN_Threshold" 2 E 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R F "" 0.35
  "$VENV_PY" "$STDP_FILE" asym sym Y 2
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax F env - Org Order $i" N 5000 "$SYN_Threshold" 2 F 20000 Y N N
  "$VENV_PY" "$SPIKES_FILE" R G "" 0.35
  "$VENV_PY" "$STDP_FILE" asym sym Y 2
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax G env - Org Order $i" N 5000 "$SYN_Threshold" 2 G 20000 Y N N
  "$VENV_PY" "$SPIKES_FILE" R H "" 0.35
  "$VENV_PY" "$STDP_FILE" asym sym Y 2
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax H env - Org Order $i" N 5000 "$SYN_Threshold" 2 H 20000 Y N N
  "$VENV_PY" "$SPIKES_FILE" R I "" 0.35
  "$VENV_PY" "$STDP_FILE" asym sym Y 2
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax I env - Org Order $i" N 5000 "$SYN_Threshold" 2 I 20000 Y N N
  
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold No PC plast E env random order $i" N 5000 "$SYN_Threshold" 2 E 5000 Y N

  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmaxpreprocess mult env A $i" Y 5000 "$SYN_Threshold" 1 A 1000 N N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax preprocess mult env B $i" Y 5000 "$SYN_Threshold" 1 B 1000 N N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax preprocess mult env E $i" Y 5000 "$SYN_Threshold" 1 E 1000 N N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax preprocess mult env F $i" Y 5000 "$SYN_Threshold" 1 F 1000 N N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax preprocess mult env G $i" Y 5000 "$SYN_Threshold" 1 G 1000 N N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax preprocess mult env H $i" Y 5000 "$SYN_Threshold" 1 H 1000 N N N
  "$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold .35 pc ratio 5 wmax preprocess mult env I $i" Y 5000 "$SYN_Threshold" 1 I 1000 N N N
  
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
  SYN_Threshold=$("$VENV_PY" -c 'import sys; print(f"{float(sys.argv[1]) + 0.25:.2f}")' "$SYN_Threshold")
done
