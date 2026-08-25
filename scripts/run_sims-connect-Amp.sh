#!/usr/bin/env bash
set -euo pipefail

# --- Paths (WSL) ---
REPO="$HOME/projects/ca3net"
VENV_PY="$REPO/.venv/bin/python"

#PYTHON_FILE="$REPO/scripts/replay_stats_simulations-param-sweep.py"
#PYTHON_FILE="$REPO/scripts/replay_stats_simulations-inhibitory-plasticity.py"
PYTHON_FILE="$REPO/scripts/replay_stats_simulations-Connect-Amp.py"
STDP_FILE="$REPO/scripts/stdp-Connect-Amp.py"
#STDP_FILE="$REPO/scripts/stdp-Inh.py"
#STDP_FILE="$REPO/scripts/stdp.py"
SPIKES_FILE="$REPO/scripts/generate_spike_train-Omen.py"
#SPIKES_FILE="$REPO/scripts/generate_spike_train-random.py"

# --- Parameters ---
ITERATIONS=8
SYN_Threshold="0.1" # Initial synaptic threshold for compression (will be incremented in each iteration)

# generate_spike_train-Omen.py sys.argv[1:4]
SPIKE_SWITCH_A="N"
SPIKE_SWITCH_RANDOM="R"
SPIKE_PF_POSTFIX_A="A"
SPIKE_PF_POSTFIX_B="B"
SPIKE_OUTPUT_SUFFIX=""

# One shared pc_ratio for all three Python scripts.
PC_RATIO="0.5"
PC_RATIO_STEP="0.05"
PC_RATIO_ITERATIONS=6

# stdp-Connect-Amp.py sys.argv[1:6]
STDP_UNUSED="asym"
STDP_MODE="sym"
STDP_LOAD_NEW="N"
STDP_LOAD_EXISTING="N"
STDP_SELECT_CONX="1"
STDP_A_MAX="5.0"

# replay_stats_simulations-Connect-Amp.py sys.argv[1:13]
REPLAY_STDP_MODE="asym"
REPLAY_STDP_INPUT="sym"
REPLAY_CUE="N"
REPLAY_PC_INDEX="1500"
REPLAY_SELECT_CONX_A="1"
REPLAY_SELECT_CONX_RANDOM="2"
REPLAY_END_DURATION="10000"
REPLAY_SAVE_PC_WEIGHTS="N"
REPLAY_DO_NOT_SAVE="N"
REPLAY_DO_SAVE="N"
REPLAY_NO_PLAST_PC="Y"
REPLAY_PLAST_PC="N"

for ((pc_ratio_iteration=1; pc_ratio_iteration<=PC_RATIO_ITERATIONS; pc_ratio_iteration++)); do
  echo "Running pc_ratio $PC_RATIO"
  STDP_A_MAX="5.0"
  for ((i=1; i<=ITERATIONS; i++)); do
    echo "Running iteration $i"

    "$VENV_PY" "$SPIKES_FILE" "$SPIKE_SWITCH_A" "$SPIKE_PF_POSTFIX_A" "$SPIKE_OUTPUT_SUFFIX" "$PC_RATIO"
    "$VENV_PY" "$STDP_FILE" "$STDP_UNUSED" "$STDP_MODE" "$STDP_LOAD_NEW" "$STDP_SELECT_CONX" "$PC_RATIO" "$STDP_A_MAX"
    "$VENV_PY" "$PYTHON_FILE" "$REPLAY_STDP_MODE" "$REPLAY_STDP_INPUT" "PC Ratio $PC_RATIO AMAX $STDP_A_MAX - Org Order $i" "$REPLAY_CUE" "$REPLAY_PC_INDEX" "$SYN_Threshold" "$REPLAY_SELECT_CONX_A" "$SPIKE_PF_POSTFIX_A" "$REPLAY_END_DURATION" "$REPLAY_SAVE_PC_WEIGHTS" "$REPLAY_DO_NOT_SAVE" "$REPLAY_NO_PLAST_PC" "$PC_RATIO"
    "$VENV_PY" "$SPIKES_FILE" "$SPIKE_SWITCH_RANDOM" "$SPIKE_PF_POSTFIX_B" "$SPIKE_OUTPUT_SUFFIX" "$PC_RATIO"
    "$VENV_PY" "$STDP_FILE" "$STDP_UNUSED" "$STDP_MODE" "$STDP_LOAD_EXISTING" "$STDP_SELECT_CONX" "$PC_RATIO" "$STDP_A_MAX"
    "$VENV_PY" "$PYTHON_FILE" "$REPLAY_STDP_MODE" "$REPLAY_STDP_INPUT" "PC Ratio $PC_RATIO AMAX $STDP_A_MAX - Org Order $i" "$REPLAY_CUE" "$REPLAY_PC_INDEX" "$SYN_Threshold" "$REPLAY_SELECT_CONX_RANDOM" "$SPIKE_PF_POSTFIX_B" "$REPLAY_END_DURATION" "$REPLAY_SAVE_PC_WEIGHTS" "$REPLAY_DO_NOT_SAVE" "$REPLAY_PLAST_PC" "$PC_RATIO"

    # Sweep threshold (bash-safe float increment)
    STDP_A_MAX=$("$VENV_PY" -c 'import sys; print(f"{float(sys.argv[1]) - 0.25:.2f}")' "$STDP_A_MAX")
  done

  PC_RATIO=$("$VENV_PY" -c 'import sys; print(f"{float(sys.argv[1]) - float(sys.argv[2]):.2f}")' "$PC_RATIO" "$PC_RATIO_STEP")
done
