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
ITERATIONS=6
SYN_Threshold="0.1" # Initial synaptic threshold for compression (will be incremented in each iteration)

for ((i=1; i<=ITERATIONS; i++)); do
  echo "Running iteration $i"

  "$VENV_PY" "$SPIKES_FILE" N A
  "$VENV_PY" "$STDP_FILE" asym sym N 1
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC-Inh 400 BC plast A env $i" Y 5000 "$SYN_Threshold" 1 A 20000 Y N N
  "$VENV_PY" "$SPIKES_FILE" R B
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "2 Env Hom $SYN_Threshold PC plast cue B env $i" Y 5000 "$SYN_Threshold" 1 B 5000 Y N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC-Inh 400 BC plast B env random order $i" Y 5000 "$SYN_Threshold" 1 B 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R C
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "3 Env Hom $SYN_Threshold PC plast cue C env $i" Y 5000 "$SYN_Threshold" 1 C 5000 Y N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC-Inh 400 BC plast C env random order $i" Y 5000 "$SYN_Threshold" 1 C 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R D
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "4 Env Hom $SYN_Threshold PC plast cue D env $i" Y 5000 "$SYN_Threshold" 1 D 5000 Y N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC-Inh 400 BC plast D env random order $i" Y 5000 "$SYN_Threshold" 1 D 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R E
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "5 Env Hom $SYN_Threshold PC plast cue E env $i" Y 5000 "$SYN_Threshold" 1 E 5000 Y N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC-Inh 400 BC plast E env random order $i" Y 5000 "$SYN_Threshold" 1 E 20000 Y Y N
  "$VENV_PY" "$SPIKES_FILE" R F
  "$VENV_PY" "$STDP_FILE" asym sym Y 1
  "$VENV_PY" "$PYTHON_FILE" asym sym "6 Env Hom $SYN_Threshold PC plast cue F env $i" Y 5000 "$SYN_Threshold" 1 F 5000 Y N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC-Inh 400 BC plast F env random order $i" Y 5000 "$SYN_Threshold" 1 F 20000 Y N N
  #"$VENV_PY" "$SPIKES_FILE" R G
  #"$VENV_PY" "$STDP_FILE" asym sym Y 1
  #"$VENV_PY" "$PYTHON_FILE" asym sym "7 Env Hom $SYN_Threshold No plast G env $i" Y 5000 "$SYN_Threshold" 1 G 5000 Y N Y
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC-Inh 400 BC plast G env random order $i" Y 5000 "$SYN_Threshold" 1 G 20000 Y N N
  #"$VENV_PY" "$SPIKES_FILE" R H
  #"$VENV_PY" "$STDP_FILE" asym sym Y 1
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC-Inh 400 BC plast H env random order $i" Y 5000 "$SYN_Threshold" 1 H 20000 Y N N
  #"$VENV_PY" "$SPIKES_FILE" R I
  #"$VENV_PY" "$STDP_FILE" asym sym Y 1
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC-Inh 400 BC plast I env random order $i" Y 5000 "$SYN_Threshold" 1 I 20000 Y N N
  #"$VENV_PY" "$SPIKES_FILE" R J
  #"$VENV_PY" "$STDP_FILE" asym sym Y 1
  #"$VENV_PY" "$PYTHON_FILE" asym sym "Hom $SYN_Threshold Sweep PC-Inh 400 BC plast J env random order $i" Y 5000 "$SYN_Threshold" 1 J 20000 Y N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "6 Env Hom $SYN_Threshold No plast F env $i" N 5000 "$SYN_Threshold" 1 F 5000 Y N Y
  #"$VENV_PY" "$PYTHON_FILE" asym sym "10 Env Hom $SYN_Threshold PC-Inh 400 BC plast A env cue $i" Y 5000 "$SYN_Threshold" 1 A 1000 Y N N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact no plast $i" N 5000 "$SYN_Threshold" 1 A 5000 Y N Y
  #"$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 6s $i" N 5000 "$SYN_Threshold" 1 A 1000 Y Y N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 10s $i" N 5000 "$SYN_Threshold" 1 A 5000 Y Y N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 15s $i" N 5000 "$SYN_Threshold" 1 A 10000 Y Y N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 20s $i" N 5000 "$SYN_Threshold" 1 A 15000 Y Y N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 25s $i" N 5000 "$SYN_Threshold" 1 A 20000 Y Y N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 30s $i" N 5000 "$SYN_Threshold" 1 A 25000 Y Y N
  #"$VENV_PY" "$PYTHON_FILE" asym sym "PC plast weights impact plast 60s $i" N 5000 "$SYN_Threshold" 1 A 55000 Y Y N

    # Sweep threshold (bash-safe float increment)
  #SYN_Threshold=$("$VENV_PY" -c 'import sys; print(f"{float(sys.argv[1]) + 0.25:.2f}")' "$SYN_Threshold")
done
