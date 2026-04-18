# -*- coding: utf8 -*-
"""
Generates hippocampal like spike trains (see also helper file: `poisson_proc.py`)
authors: András Ecker, Eszter Vértes, Szabolcs Káli last update: 10.2018
"""
import sys, warnings
import os, pickle
import numpy as np
from tqdm import tqdm  # progress bar
from poisson_proc import hom_poisson, inhom_poisson
from helper import save_place_fields, refractoriness 


base_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

outfield_rate = 0.1  # avg. firing rate outside place field [Hz]
infield_rate = 20.0  # avg. in-field firing rate [Hz]
t_max = 405.0  # [s]

def swap_spike_train_segments(spike_trains, start_a, start_b, seg_len):
    """
    Swap spike train patterns between two contiguous neuron-ID segments:
    [start_a, start_a+seg_len) <-> [start_b, start_b+seg_len)

    spike_trains: list-like of length n_neurons, where spike_trains[i] are times for neuron i
    """
    n = len(spike_trains)

    if seg_len <= 0:
        return spike_trains  # no-op

    end_a = start_a + seg_len
    end_b = start_b + seg_len

    if not (0 <= start_a < n and 0 <= start_b < n):
        raise ValueError(f"Segment starts must be in [0, {n-1}]")

    if end_a > n or end_b > n:
        raise ValueError(f"Segments exceed neuron range: n={n}, "
                         f"A=[{start_a},{end_a}), B=[{start_b},{end_b})")

    # Optional: prevent overlapping segments (ambiguous swap semantics)
    if not (end_a <= start_b or end_b <= start_a):
        raise ValueError("Swap segments overlap; choose non-overlapping ranges.")

    # Do the swap (copy slice to avoid aliasing)
    tmp = spike_trains[start_a:end_a]
    spike_trains[start_a:end_a] = spike_trains[start_b:end_b]
    spike_trains[start_b:end_b] = tmp

    return spike_trains



def generate_spike_train(n_neurons, place_cell_ratio, linear, ordered=True, seed=1234, swap_start_a=None, swap_start_b=None, swap_len=0, PF_pklf_postfix=None):
    assert n_neurons >= 1000, "Assumptions hold only for a reasonably big group of neurons"
    assert 0.0 < place_cell_ratio <= 1.0

    neuronIDs = np.arange(n_neurons)
    rng = np.random.default_rng(seed)

    # --- Choose place cells
    if linear:
        # Oversample both ends (first/last 100)
        p_uniform = 1.0 / n_neurons
        tmp = (1.0 - 2 * 2 * 100 * p_uniform) / (n_neurons - 200)
        assert tmp > 0, "Invalid sampling distribution; increase n_neurons or change end oversampling."

        p = np.concatenate([
            2 * p_uniform * np.ones(100),
            tmp * np.ones(n_neurons - 200),
            2 * p_uniform * np.ones(100),
        ])
        p= (1.0 / n_neurons) * np.ones(n_neurons)  # Uniform sampling (comment out to use oversampling of ends)
        place_cells = np.sort(
            rng.choice(neuronIDs, int(n_neurons * place_cell_ratio), replace=False, p=p),
            kind="mergesort"
        )
    else:
        place_cells = np.sort(
            rng.choice(neuronIDs, int(n_neurons * place_cell_ratio), replace=False),
            kind="mergsort"
        )

    place_cells_set = set(place_cells)

    # --- PF starts
    if ordered:
        # Keep your original "ordered" behavior as close as possible:
        # draw n_neurons, sort, then index by place_cells
        phi_pool = np.sort(rng.random(n_neurons), kind="mergesort")
        phi_starts = phi_pool[place_cells] * 2 * np.pi
    else:
        # PF starts drawn for place cells then sorted
        phi_starts = np.sort(rng.random(len(place_cells)), kind="mergesort") * 2 * np.pi

    if linear:
        phi_starts = phi_starts - 0.1 * np.pi
        if PF_pklf_postfix is not None:
            pklf_name = os.path.join(base_path, "files",
                                     f"PFstarts_{place_cell_ratio}_linear{'_no' if not ordered else ''}_{PF_pklf_postfix}.pkl")
        else:
            pklf_name = os.path.join(base_path, "files",
                                 f"PFstarts_{place_cell_ratio}_linear{'_no' if not ordered else ''}.pkl")
    else:
        pklf_name = os.path.join(base_path, "files",
                                 f"PFstarts_{place_cell_ratio}{'_no' if not ordered else ''}.pkl")

    place_fields = {int(neuron_id): float(phi_starts[i]) for i, neuron_id in enumerate(place_cells)}
    save_place_fields(place_fields, pklf_name)

    # --- Generate per-neuron seeds (so we never touch global np.random.seed)
    # Use uint32 range to be safe for libraries expecting 32-bit seeds
    per_neuron_seeds = rng.integers(0, 2**32 - 1, size=n_neurons, dtype=np.uint32)

    spike_trains = []
    for neuron_id in tqdm(range(n_neurons)):
        s = int(per_neuron_seeds[neuron_id])

        if neuron_id in place_cells_set:
            spike_train = inhom_poisson(infield_rate, t_max, place_fields[neuron_id], linear, s)
        else:
            spike_train = hom_poisson(outfield_rate, 100, t_max, s)

        spike_trains.append(spike_train)
    if swap_start_a is not None and swap_start_b is not None and swap_len > 0:
        spike_trains = swap_spike_train_segments(spike_trains, swap_start_a, swap_start_b, swap_len)

    return spike_trains



if __name__ == "__main__":
    try:
        SwitchSection = sys.argv[1]
        PF_pklf_postfix = sys.argv[2] if len(sys.argv) > 2 else None
    except:
        SwitchSection = None
        PF_pklf_postfix = None
    # --- Parameters
    n_neurons = 8000
    place_cell_ratio = 0.5
    linear = True
    f_out = "spike_trains_%.1f_linear.npz"%place_cell_ratio if linear else "spike_trains_%.1f.npz"%place_cell_ratio; ordered = True
    #f_out = "intermediate_spike_trains_%.1f_linear.npz"%place_cell_ratio if linear else "intermediate_spike_trains_%.1f.npz"%place_cell_ratio; ordered = False

    
    if SwitchSection == "Y":
        spike_trains = generate_spike_train(n_neurons, place_cell_ratio, linear=linear, ordered=ordered,swap_start_a=1500, swap_start_b=4500, swap_len=1000, PF_pklf_postfix=PF_pklf_postfix)
    else:
        spike_trains = generate_spike_train(n_neurons, place_cell_ratio, linear=linear, ordered=ordered, PF_pklf_postfix=PF_pklf_postfix)
    spike_trains = refractoriness(spike_trains)  # clean spike train (based on refractory period)
    spike_trains = np.array(spike_trains, dtype=object)
    npzf_name = os.path.join(base_path, "files", f_out)
    np.savez(npzf_name, spike_trains=spike_trains)
