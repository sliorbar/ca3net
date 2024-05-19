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
set_device("cpp_standalone")  # speed up the simulation with generated C++ code
import matplotlib.pyplot as plt
from helper import load_spike_trains, save_wmx
from plots import plot_STDP_rule, plot_wmx, plot_wmx_avg, plot_w_distr, save_selected_w, plot_weights


warnings.filterwarnings("ignore")
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
connection_prob_PC = 0.1
nPCs = 8000

## Traing with inhibitory neurons
max_syn_strength = 3.0  
stdp_pre_scale_factor_bc_i = 1.0
stdp_post_scale_factor_bc_i = 1.0
stdp_pre_scale_factor_bc_e = 1.0
stdp_post_scale_factor_bc_e = 1.0
stdp_pre_scale_factor_pc_i = 1.0
stdp_post_scale_factor_pc_i = 1.0
# Include additional BC during training
nBCs = 150
connection_prob_BC = 0.25
rise_PC_I = 0.3 * ms  # Bartos 2002 (20-80%)
rise_BC_E = 1. * ms  # Lee 2014 (data from CA1)
rise_BC_I = 0.25 * ms  # Bartos 2002 (20-80%)
decay_PC_I = 3.3 * ms  # Bartos 2002
decay_BC_E = 4.1 * ms  # Lee 2014 (data from CA1)
decay_BC_I = 1.2 * ms  # Bartos 2002
# Normalization factors (normalize the peak of the PSC curve to 1)
tp = (decay_PC_I * rise_PC_I)/(decay_PC_I - rise_PC_I) * np.log(decay_PC_I/rise_PC_I)
norm_PC_I = 1.0 / (np.exp(-tp/decay_PC_I) - np.exp(-tp/rise_PC_I))
tp = (decay_BC_E * rise_BC_E)/(decay_BC_E - rise_BC_E) * np.log(decay_BC_E/rise_BC_E)
norm_BC_E = 1.0 / (np.exp(-tp/decay_BC_E) - np.exp(-tp/rise_BC_E))
tp = (decay_BC_I * rise_BC_I)/(decay_BC_I - rise_BC_I) * np.log(decay_BC_I/rise_BC_I)
norm_BC_I = 1.0 / (np.exp(-tp/decay_BC_I) - np.exp(-tp/rise_BC_I))
delay_PC_I = 1.1 * ms  # Bartos 2002
delay_BC_E = 0.9 * ms  # Geiger 1997 (data from DG)
delay_BC_I = 0.6 * ms  # Bartos 2002

Erev_E = 0.0 * mV
Erev_I = -70.0 * mV

rate_MF = 15.0 * Hz  # mossy fiber input freq

z = 1 * nS
# parameters for BCs (re-optimized by Szabolcs)
g_leak_BC = 7.51454086502288 * nS
tau_mem_BC = 15.773412296065 * ms
Cm_BC = tau_mem_BC * g_leak_BC
Vrest_BC = -74.74167987795019 * mV
Vreset_BC = -64.99190523539687 * mV
theta_BC = -57.7092044103536 * mV
tref_BC = 1.15622717832178 * ms
delta_T_BC = 4.58413312063091 * mV
spike_th_BC = theta_BC + 5 * delta_T_BC
a_BC = 3.05640210724374 * nS
b_BC = 0.916098931234532 * pA
#a_BC = 0 * nS
#b_BC = 0 * pA

tau_w_BC = 178.581099914024 * ms
taup_sim_bc_i = 20.0
taum_sim_bc_i = 20.0
stdp_post_scale_factor_bc_i = 0.5 # Post before pre factor - Positive number is LTD
stdp_pre_scale_factor_bc_i = 0.5    #Use to modify the pre / post window = Positive number is LTP
Learning_Rate = 0.01
eqs_BC = """
dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm- theta_BC)/delta_T_BC) - w - (g_ampa*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_BC : volt (unless refractory)
dw/dt = (a_BC*(vm-Vrest_BC) - w) / tau_w_BC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_BC_E : 1
dx_ampa/dt = -x_ampa/decay_BC_E : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_BC_I : 1
dx_gaba/dt = -x_gaba/decay_BC_I : 1
"""




def learning(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, w_init):
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
    STDP.connect(condition="i!=j", p=connection_prob_PC)
    STDP.w = w_init
    
    # Inhibitory neurons definition
    BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    BCs.vm  = Vrest_BC; BCs.g_ampa = 0.0; BCs.g_gaba = 0.0
    
    #Synapses definition
    #There are three sybapses: BC->PC == BC_E, PC->BC == PC_I, BC->BC == BC_I
    ### Add BCs to the model
    w_BC_I = 5.0
    w_PC_I = 0.65  # nS
    w_BC_E = 0.85
    wmax_bc_i = w_BC_I * max_syn_strength
    taup_bc_i = taup_sim_bc_i * ms 
    taum_bc_i = taum_sim_bc_i * ms
    
    Ap = Learning_Rate
    Am_bc_i = -Ap * stdp_post_scale_factor_bc_i # Post syn stdp 
    Ap_bc_i = Ap * stdp_pre_scale_factor_bc_i
   
    
    Ap *= wmax_bc_i
    Am *= wmax_bc_i 
    #To align with code in Brian2 documentation (https://brian2.readthedocs.io/en/latest/examples/frompapers.Izhikevich_2007.html?highlight=stdp#example-izhikevich-2007)
    dApresyn = Ap_bc_i
    dApostsyn = Am_bc_i 

    #C_PC_E = Synapses(PCs, PCs, "w_exc:1",  on_pre="x_ampa+=0", delay=delay_PC_E)
    #C_PC_E.w_exc = C_PC_E.w_exc*1e-9 #Convert back to ns
    synapse_setup_bc_i='''
    w_inh:1
    dApresyn_bc_i/dt = -Apresyn_bc_i/taup_bc_i : 1 (event-driven)
    dApostsyn_bc_i/dt = -Apostsyn_bc_i/taum_bc_i : 1 (event-driven)
    '''
    #(event-driven)
    on_pre_setup_bc_i = '''
    x_gaba+=norm_BC_I*w_inh
    Apresyn_bc_i += dApresyn_bc_i
    w_exc = clip(w_inh + Apostsyn_bc_i,0,wmax_bc_i)
    '''
    on_post_setup_bc_i= '''
    Apostsyn_bc_I += dApostsyn_bc_i
    w_inh = clip(w_inh + Apresyn_bc_i,0,wmax_bc_i)
    '''
    #C_PC_E_STDP = Synapses(PCs, PCs,synapse_setup, on_pre=on_pre_setup,on_post=on_post_setup,delay=delay_PC_E)

    C_PC_I = Synapses(BCs, PC, on_pre="x_gaba+=norm_PC_I*w_PC_I", delay=delay_PC_I)
    C_PC_I.connect(p=connection_prob_BC)

    C_BC_E = Synapses(PC, BCs, on_pre="x_ampa+=norm_BC_E*w_BC_E", delay=delay_BC_E)
    C_BC_E.connect(p=connection_prob_PC)

    C_BC_I = Synapses(BCs, BCs, on_pre="x_gaba+=norm_BC_I*w_BC_I", delay=delay_BC_I)
    C_BC_I.connect(condition="i!=j",p=connection_prob_BC)

    SM_PC = SpikeMonitor(PC)
    SM_BC = SpikeMonitor(BCs)
    RM_PC = PopulationRateMonitor(PC)
    RM_BC = PopulationRateMonitor(BCs)




    ###
    run(400*second, report="text")
    
    weightmx = np.zeros((nPCs, nPCs))
    weightmx[STDP.i[:], STDP.j[:]] = STDP.w[:]
    return weightmx * 1e9  # *1e9 nS conversion


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[2]
    except:
        STDP_mode = "sym"
    assert STDP_mode in ["asym", "sym"]

    place_cell_ratio = 0.5
    linear = True
    f_in = "spike_trains_%.1f_linear.npz" % place_cell_ratio if linear else "spike_trains_%.1f.npz" % place_cell_ratio
    f_out = "lb_wmx_%s_%.1f_linear.npz" % (STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)
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

    weightmx = learning(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, w_init)
    weightmx *= scale_factor  # quick and dirty additional scaling! (in an ideal world the STDP parameters should be changed to include this scaling...)
    save_wmx(weightmx, os.path.join(base_path, "files", f_out))

    plot_wmx(weightmx, save_name=f_out[:-4])
    plot_wmx_avg(weightmx, n_pops=100, save_name="%s_avg" % f_out[:-4])
    plot_w_distr(weightmx, save_name="%s_distr" % f_out[:-4])
    selection = np.array([500, 2400, 4000, 5500, 7015])
    plot_weights(save_selected_w(weightmx, selection), save_name="%s_sel_weights" % f_out[:-4])
    device.delete()
    plt.show()
