# -*- coding: utf8 -*-
"""
Loads in hippocampal like spike train (produced by `generate_spike_train.py`) and runs STD learning rule in a recurrent spiking neuron population
-> creates weight matrix for PC population, used by `spw*` scripts
updated to produce symmetric STDP curve as reported in Mishra et al. 2016 - 10.1038/ncomms11552
authors: András Ecker, Eszter Vértes, last update: 11.2017
"""

import os, sys, warnings
import numpy as np
import random as pyrandom
from brian2 import *
from brian2.units.allunits import *
from brian2.units.stdunits import *
from scipy.sparse import coo_matrix
import scipy.sparse
#set_device("cpp_standalone", build_on_run=False)  # speed up the simulation with generated C++ code
set_device("cpp_standalone")
import matplotlib.pyplot as plt
from helper import load_spike_trains, save_wmx,load_wmx
from plots import plot_STDP_rule, plot_wmx, plot_wmx_avg, plot_w_distr, save_selected_w, plot_weights


warnings.filterwarnings("ignore")
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
connection_prob_PC = 0.1
nPCs = 8000


def learning(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, w_init, w_exc = None):
    """
    Takes a spiking group of neurons, connects the neurons sparsely with each other, and learns the weight 'pattern' via STDP:
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param spiking_neurons, spike_times: np.arrays for Brian2's SpikeGeneratorGroup (list of lists created by `generate_spike_train.py`) - spike train used for learning
    :param taup, taum: time constant of weight change (in ms)
    :param Ap, Am: max amplitude of weight change
    :param wmax: maximum weight (in S)
    :param w_init: initial weights (in S)
    :return weightmx: learned synaptic weights
    """

    np.random.seed(12345)
    pyrandom.seed(12345)
    #plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, "STDP_rule")

    PC = SpikeGeneratorGroup(nPCs, spiking_neurons, spike_times*second)
    # mimics Brian1's exponentialSTPD class, with interactions='all', update='additive'
    # see more on conversion: http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html
    STDP = Synapses(PC, PC,
            """
            w : 1
            dA_presyn/dt = -A_presyn/taup : 1 (event-driven)
            dA_postsyn/dt = -A_postsyn/taum : 1 (event-driven)
            """,
            on_pre="""
            A_presyn += Ap
            w = clip(w + A_postsyn, 0, wmax)
            """,
            on_post="""
            A_postsyn += Am
            w = clip(w + A_presyn, 0, wmax)
            """)
    
    if  w_exc == None:
        STDP.connect(condition="i!=j", p=connection_prob_PC)
        STDP.w = w_init
    else:
        STDP.connect(i=w_exc.row, j=w_exc.col)
        STDP.w = w_exc.data
        #STDP.w = w_init
    #net = Network(PC,STDP)
    #net.run(200 * second, report="text")
    #device.delete()
    run(300 * second, report="text")
    #device.build(directory='output', compile=True, run=True, debug=False)
    weightmx = np.zeros((nPCs, nPCs))
    weightmx[STDP.i[:], STDP.j[:]] = STDP.w[:]
    return weightmx * 1e9  # *1e9 nS conversion


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[2]
        STDP_mode_Input = sys.argv[1]
    except:
        STDP_mode = "sym"
    assert STDP_mode in ["asym", "sym"]

    place_cell_ratio = 0.5
    linear = True
    f_in = "spike_trains_%.1f_linear.npz" % place_cell_ratio if linear else "spike_trains_%.1f.npz" % place_cell_ratio
    #f_out = "wmx_%s_%.1f_linear-itr.npz" % (STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)
    f_out = "wmx_%s_%.1f_linear-itr300.npz" % (STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)
    #f_in = "intermediate_spike_trains_%.1f_linear.npz" % place_cell_ratio if linear else "intermediate_spike_trains_%.1f.npz" % place_cell_ratio
    #f_out = "intermediate_wmx_%s_%.1f_linear.npz" % (STDP_mode, place_cell_ratio) if linear else "intermediate_wmx_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)

    # STDP parameters (see `optimization/analyse_STDP.py`)
    if STDP_mode == "asym":
        taup = taum = 20 * ms
        Ap = 0.01
        Am = -Ap
        wmax = 4e-8  # S
        scale_factor = 1.27
    elif STDP_mode == "sym":
        taup = taum = 62.5 * ms
        Ap = Am = 4e-3
        wmax = 2e-8  # S
        scale_factor = 0.62
    w_init = 1e-10  # S
    Ap *= wmax; Am *= wmax  # needed to reproduce Brian1 results

    spiking_neurons, spike_times = load_spike_trains(os.path.join(base_path, "files", f_in))
    PF_pklf_name = os.path.join(base_path, "files", "PFstarts_%s_linear_itr.pkl" % place_cell_ratio) if linear else None
    try:
        f_in = "wmx_after_run_%s_%.1f_linear-itr20ms.npz" % (STDP_mode_Input, place_cell_ratio) if linear else "wmx_after_run_%s_%.1f.pkl" % (STDP_mode_Input, place_cell_ratio)
        #f_in = "wmx_%s_%.1f_linear.npz" % (STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)
         
        wmx_PC_E = load_wmx(os.path.join(base_path, "files", f_in))  # *1e9 nS conversion
        #wmx_PC_E = wmx_PC_E / (scale_factor * 1e9)
        wmx_PC_E = wmx_PC_E / (1e9)
        #dense = wmx_PC_E.todense()
        #wmx_PC_E = dense
    except:
        wmx_PC_E = None
    

    weightmx = learning(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, w_init,w_exc=wmx_PC_E)
    weightmx *= scale_factor  # quick and dirty additional scaling! (in an ideal world the STDP parameters should be changed to include this scaling...)
    save_wmx(weightmx, os.path.join(base_path, "files", f_out))

    plot_wmx(weightmx, save_name=f_out[:-4])
    #plot_wmx_avg(weightmx, n_pops=100, save_name="%s_avg" % f_out[:-4])
    #plot_w_distr(weightmx, save_name="%s_distr" % f_out[:-4])
    selection = np.array([500, 2400, 4000, 5500, 7015])
    plot_weights(save_selected_w(weightmx, selection), save_name="%s_sel_weights" % f_out[:-4])
    plt.show()
