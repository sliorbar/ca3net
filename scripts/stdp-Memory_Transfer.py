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
import random 

warnings.filterwarnings("ignore")
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])

nPCs = 8000
nBCs = 150
plasticity_scale_factor = 0.5  # scaling factor for the STDP window 
# sparseness
connection_prob_PC = 0.1
connection_prob_BC_E = 0.1
connection_prob_BC = 0.25
connection_prob_CA1_PC = 0.05
connection_prob_CA1_BC = 0.1
rise_PC_E = 1.3 * ms  # Guzman 2016 (only from Fig.1 H - 20-80%)
rise_PC_MF = 0.65 * ms  # Vyleta ... Jonas 2016 (20-80%)
rise_PC_I = 0.3 * ms  # Bartos 2002 (20-80%)
rise_BC_E = 1. * ms  # Lee 2014 (data from CA1)
rise_BC_I = 0.25 * ms  # Bartos 2002 (20-80%)
# decay time constants
decay_PC_E = 9.5 * ms  # Guzman 2016 ("needed for temporal summation of EPSPs")
decay_PC_MF = 5.4 * ms  # Vyleta ... Jonas 2016
decay_PC_I = 3.3 * ms  # Bartos 2002
decay_BC_E = 4.1 * ms  # Lee 2014 (data from CA1)
decay_BC_I = 1.2 * ms  # Bartos 2002
# Normalization factors (normalize the peak of the PSC curve to 1)
tp = (decay_PC_E * rise_PC_E)/(decay_PC_E - rise_PC_E) * np.log(decay_PC_E/rise_PC_E)  # time to peak
norm_PC_E = 1.0 / (np.exp(-tp/decay_PC_E) - np.exp(-tp/rise_PC_E))
tp = (decay_PC_MF * rise_PC_MF)/(decay_PC_MF - rise_PC_MF) * np.log(decay_PC_MF/rise_PC_MF)
norm_PC_MF = 1.0 / (np.exp(-tp/decay_PC_MF) - np.exp(-tp/rise_PC_MF))
tp = (decay_PC_I * rise_PC_I)/(decay_PC_I - rise_PC_I) * np.log(decay_PC_I/rise_PC_I)
norm_PC_I = 1.0 / (np.exp(-tp/decay_PC_I) - np.exp(-tp/rise_PC_I))
tp = (decay_BC_E * rise_BC_E)/(decay_BC_E - rise_BC_E) * np.log(decay_BC_E/rise_BC_E)
norm_BC_E = 1.0 / (np.exp(-tp/decay_BC_E) - np.exp(-tp/rise_BC_E))
tp = (decay_BC_I * rise_BC_I)/(decay_BC_I - rise_BC_I) * np.log(decay_BC_I/rise_BC_I)
norm_BC_I = 1.0 / (np.exp(-tp/decay_BC_I) - np.exp(-tp/rise_BC_I))

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

delay_PC_I = 1.1 * ms  # Bartos 2002
delay_BC_E = 0.9 * ms  # Geiger 1997 (data from DG)
delay_BC_I = 0.6 * ms  # Bartos 2002
'''Modify the code to remove synaptic delay'''
z = 1 * nS
wmax = 2e-8  # S
# synaptic reversal potentials
Erev_E = 0.0 * mV
Erev_I = -70.0 * mV

tau_w_BC = 178.581099914024 * ms
eqs_BC = """
dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm- theta_BC)/delta_T_BC) - w - (g_ampa*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_BC : volt (unless refractory)
dw/dt = (a_BC*(vm-Vrest_BC) - w) / tau_w_BC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_BC_E : 1
dx_ampa/dt = -x_ampa/decay_BC_E : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_BC_I : 1
dx_gaba/dt = -x_gaba/decay_BC_I : 1
"""


def learning(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, w_init, fin = None, LoadMatrix = None,
             spiking_neurons_CA1 = None, spike_times_CA1 = None, sim_duration = 400):
    """
    Takes a spiking group of neurons, connects the neurons sparsely with each other, and learns the weight 'pattern' via STDP:
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param spiking_neurons, spike_times: np.arrays for Brian2's SpikeGeneratorGroup (list of lists created by `generate_spike_train.py`) - spike train used for learning (CA3/PC population)
    :param spiking_neurons_CA1, spike_times_CA1: same as above but for the CA1 population; falls back to the CA3 spike train if not given
    :param taup, taum: time constant of weight change (in ms)
    :param Ap, Am: max amplitude of weight change
    :param wmax: maximum weight (in S)
    :param w_init: initial weights (in S)
    :return weightmx: learned synaptic weights
    """

    if spiking_neurons_CA1 is None or spike_times_CA1 is None:
        spiking_neurons_CA1, spike_times_CA1 = spiking_neurons, spike_times

    np.random.seed(12345)
    pyrandom.seed(12345)
    #plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, "STDP_rule")
    max_mult = 1.5  # Allow for maximum 1.5x scaling of the Ph2 weight
    max_mult_BC_E = 1.5  # Allow for maximum 1.5x scaling of the Ph2 weight
    initial_mult = 1.0  # Initial scaling of the weight - 50%
    step_size_range = (0.01, 0.03)  # Range for random step size selection, in nS. Set to a narrow range to ensure reproducibility while allowing for some variability in the results.
    step_size = random.uniform(*step_size_range)  # Randomly select step size from the specified range
    #step_size = 0.01
    inh_tau = 40 * ms
    w_PC_I_inp = 0.65 #* 1e-9 # nS # Taken from Ecker 2022
    w_BC_E_inp = 0.85 #* 1e-9 # nS # Taken from Ecker 2022
    w_BC_I_inp = 5.0 #* 1e-9 # nS # Taken from Ecker 2022
    w_CA1_PC_inp = 0.1 #* 1e-9 # nS # Taken from Ecker 2022
    w_CA1_BC_inp = 1.0 #* 1e-9 # nS # Taken from Ecker 2022
    wmax_PC_I = 1.2 # nS Allow for maximum  scaling of the weight
    wmax_BC_E = 1.8 # nS Allow for maximum  scaling of the weight
    wmax_BC_I = 10.0 # nS Allow for maximum scaling of the weight
    wmax_CA1_PC = 1.5 # nS Allow for maximum scaling of the weight
    wmax_CA1_BC = 5 # nS Allow for maximum scaling of the weight

    PC = SpikeGeneratorGroup(nPCs, spiking_neurons, spike_times*second)
   
    # mimics Brian1's exponential STPD class, with interactions='all', update='additive'
    # see more on conversion: http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html
    
    # Inhinitory population
    BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    BCs.vm  = Vrest_BC; BCs.g_ampa = 0.0; BCs.g_gaba = 0.0    

    #CA1 Area population
    CA1_PCs = SpikeGeneratorGroup(nPCs, spiking_neurons_CA1, spike_times_CA1*second)
    CA1_BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    CA1_BCs.vm  = Vrest_BC; CA1_BCs.g_ampa = 0.0; CA1_BCs.g_gaba = 0.0 

    #PC to BC plasticity parameters (Ap > 0 is hSTDP)
    Ap_PC_I = -step_size
    Am_PC_I = -Ap_PC_I
    # BC_E plasticity parameters (Ap > 0 is hSTDP)
    Ap_BC_E = -step_size 
    Am_BC_E = -Ap_BC_E
    # BC_I plasticity parameters (Ap > 0 is hSTDP)
    Ap_BC_I = step_size
    Am_BC_I = Ap_BC_I
    # Scale the plasticity parameters to match the weight range
    # CA1 plasticity parameters (Ap > 0 is hSTDP)
    Ap_CA1_PC = step_size
    Am_CA1_PC = Ap_CA1_PC
    Ap_CA1_BC = -step_size
    Am_CA1_BC = Ap_CA1_BC



    Ap_PC_I = wmax_PC_I * Ap_PC_I
    Am_PC_I = wmax_PC_I * Am_PC_I
    Ap_BC_E = wmax_BC_E * Ap_BC_E
    Am_BC_E = wmax_BC_E * Am_BC_E
    Ap_BC_I = wmax_BC_I * Ap_BC_I
    Am_BC_I = wmax_BC_I * Am_BC_I
    Ap_CA1_PC = wmax_CA1_PC * Ap_CA1_PC
    Am_CA1_PC = wmax_CA1_PC * Am_CA1_PC
    Ap_CA1_BC = wmax_CA1_BC * Ap_CA1_BC
    Am_CA1_BC = wmax_CA1_BC * Am_CA1_BC

    dApresyn = Ap
    dApostsyn = Am
    dApresyn_BC_I = Ap_BC_I
    dApostsyn_BC_I = Am_BC_I
    dApresyn_BC_E = Ap_BC_E 
    dApostsyn_BC_E = Am_BC_E
    dApresyn_PC_I = Ap_PC_I
    dApostsyn_PC_I = Am_PC_I
    dApresyn_CA1_PC = Ap_CA1_PC
    dApostsyn_CA1_PC = Am_CA1_PC
    dApresyn_CA1_BC = Ap_CA1_BC
    dApostsyn_CA1_BC = Am_CA1_BC

    tau_bc = inh_tau  # Different tau for BCs, as in Bartos 2002
    # PC_I modeling
    synapse_model_PC_I='''
    w_e_inh:1
    dApresyn_PC_I/dt = -Apresyn_PC_I/tau_bc : 1 (event-driven)
    dApostsyn_PC_I/dt = -Apostsyn_PC_I/tau_bc : 1 (event-driven)
    '''
    on_pre_setup_PC_I = '''
    x_ampa+=norm_PC_I*w_e_inh
    Apresyn_PC_I += dApresyn_PC_I
    w_e_inh = clip(w_e_inh + Apostsyn_PC_I,0,wmax_PC_I)
    '''
    on_post_setup_PC_I= '''
    Apostsyn_PC_I += dApostsyn_PC_I
    w_e_inh = clip(w_e_inh + Apresyn_PC_I,0,wmax_PC_I)
    '''
    # BC_E modeling
    synapse_model_BC_E='''
    w_i_exc:1
    dApresyn_BC_E/dt = -Apresyn_BC_E/taup : 1 (event-driven)
    dApostsyn_BC_E/dt = -Apostsyn_BC_E/taum : 1 (event-driven)
    '''
    on_pre_setup_BC_E = '''
    Apresyn_BC_E += dApresyn_BC_E
    w_i_exc = clip(w_i_exc + Apostsyn_BC_E,0,wmax_BC_E)
    '''
    on_post_setup_BC_E= '''
    Apostsyn_BC_E += dApostsyn_BC_E
    w_i_exc = clip(w_i_exc + Apresyn_BC_E,0,wmax_BC_E)
    '''
    # BC_I modeling
    synapse_model_BC_I='''
    w_i_inh:1
    dApresyn_BC_I/dt = -Apresyn_BC_I/tau_bc : 1 (event-driven)
    dApostsyn_BC_I/dt = -Apostsyn_BC_I/tau_bc : 1 (event-driven)
    '''
    on_pre_setup_BC_I = '''
    x_gaba+=norm_BC_I*w_i_inh
    Apresyn_BC_I += dApresyn_BC_I
    w_i_inh = clip(w_i_inh + Apostsyn_BC_I,0,wmax_BC_I)
    '''
    on_post_setup_BC_I= '''
    Apostsyn_BC_I += dApostsyn_BC_I
    w_i_inh = clip(w_i_inh + Apresyn_BC_I,0,wmax_BC_I)
    '''
    # Synapses definition
    #PC to PC STDP
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
    #STDP.delay = 0.1 * ms  # Bartos 2002
    STDP.w = w_init
    
    # PC to BC synapses       
    if LoadMatrix != None:
        wmx_PC_I = np.load(fin[:-4] + "_PC_I.npz")
        wmx_BC_E = np.load(fin[:-4] + "_BC_E.npz")
        wmx_BC_I = np.load(fin[:-4] + "_BC_I.npz")
    C_PC_I = Synapses(source=PC, target=BCs, model=synapse_model_PC_I, on_pre= on_pre_setup_PC_I, on_post= on_post_setup_PC_I, delay=delay_PC_I)
    if LoadMatrix != None:
        C_PC_I.connect(i=wmx_PC_I.row, j=wmx_PC_I.col)
        C_PC_I.w_e_inh = wmx_PC_I.data
    else:
        C_PC_I.connect(p=connection_prob_BC)
        C_PC_I.w_e_inh = w_PC_I_inp
    # BC to PC 
    C_BC_E = Synapses(source=BCs, target=PC,model=synapse_model_BC_E, on_pre=on_pre_setup_BC_E, on_post=on_post_setup_BC_E , delay=delay_BC_E)
    
    if LoadMatrix != None:
        C_BC_E.connect(i=wmx_BC_E.row, j=wmx_BC_E.col)
        C_BC_E.w_i_exc = wmx_BC_E.data
    else:
        C_BC_E.connect(p=connection_prob_BC_E)
        C_BC_E.w_i_exc = w_BC_E_inp
    # BC to BC
    C_BC_I = Synapses(source=BCs, target=BCs,model=synapse_model_BC_I , on_pre=on_pre_setup_BC_I, on_post= on_post_setup_BC_I, delay=delay_BC_I)
    
    if LoadMatrix != None:
        C_BC_I.connect(i=wmx_BC_I.row, j=wmx_BC_I.col)
        C_BC_I.w_i_inh = wmx_BC_I.data
    else:
        C_BC_I.connect(p=connection_prob_BC)
        C_BC_I.w_i_inh = w_BC_I_inp
    # CA1 to PC
    delay_CA1_PC = 5.0 * ms  # 
    delay_CA1_BC = 5.0 * ms  #
    
    C_CA1_PC = Synapses(PC, CA1_PCs,
            """
            w_exc : 1
            dApresyn_CA1_PC/dt = -Apresyn_CA1_PC/taup : 1 (event-driven)
            dApostsyn_CA1_PC/dt = -Apostsyn_CA1_PC/taum : 1 (event-driven)
            """,
            on_pre="""
            Apresyn_CA1_PC += Ap_CA1_PC
            w_exc = clip(w_exc + Apostsyn_CA1_PC, 0, wmax_CA1_PC)
            """,
            on_post="""
            Apostsyn_CA1_PC += Am_CA1_PC
            w_exc = clip(w_exc + Apresyn_CA1_PC, 0, wmax_CA1_PC)
            """, delay=delay_CA1_PC)
    C_CA1_PC.connect(p=connection_prob_CA1_PC)
    #STDP.delay = 0.1 * ms  # Bartos 2002
    C_CA1_PC.w_exc = w_CA1_PC_inp

    C_CA1_BC = Synapses(PC, CA1_BCs,
            """
            w_exc : 1
            dApresyn_CA1_BC/dt = -Apresyn_CA1_BC/inh_tau : 1 (event-driven)
            dApostsyn_CA1_BC/dt = -Apostsyn_CA1_BC/inh_tau : 1 (event-driven)  
            """,
            on_pre="""
            x_ampa+=norm_PC_I * w_exc
            Apresyn_CA1_BC += Ap_CA1_BC
            w_exc = clip(w_exc + Apostsyn_CA1_BC, 0, wmax_CA1_BC)
            """,
            on_post="""
            Apostsyn_CA1_BC += Am_CA1_BC
            w_exc = clip(w_exc + Apresyn_CA1_BC, 0, wmax_CA1_BC)
            """, delay=delay_CA1_BC)
    C_CA1_BC.connect(p=connection_prob_CA1_BC)
    C_CA1_BC.w_exc = w_CA1_BC_inp
    
    
    #Run the simulation
    SM_PC = SpikeMonitor(PC)
    SM_BC = SpikeMonitor(BCs)
    SM_CA1_PC = SpikeMonitor(CA1_PCs)
    SM_CA1_BC = SpikeMonitor(CA1_BCs)
    run(sim_duration*second, report="text")
    print("Total spikes - SM_PC:", SM_PC.num_spikes)
    print("Total spikes - SM_BC:", SM_BC.num_spikes)
    print("Total spikes - SM_CA1_PC:", SM_CA1_PC.num_spikes)
    print("Total spikes - SM_CA1_BC:", SM_CA1_BC.num_spikes)
    
    weightmx = np.zeros((nPCs, nPCs))
    weightmx[STDP.i[:], STDP.j[:]] = STDP.w[:]

    weightmx_PC_I = np.zeros((nPCs, nBCs))
    weightmx_PC_I[C_PC_I.i[:], C_PC_I.j[:]] = C_PC_I.w_e_inh[:]
    weightmx_BC_E = np.zeros((nBCs, nPCs))
    weightmx_BC_E[C_BC_E.i[:], C_BC_E.j[:]] = C_BC_E.w_i_exc[:]
    weightmx_BC_I = np.zeros((nBCs, nBCs))
    weightmx_BC_I[C_BC_I.i[:], C_BC_I.j[:]] = C_BC_I.w_i_inh[:]
    weightmx_CA1_PC = np.zeros((nPCs, nPCs))
    weightmx_CA1_PC[C_CA1_PC.i[:], C_CA1_PC.j[:]] = C_CA1_PC.w_exc[:]
    weightmx_CA1_BC = np.zeros((nPCs, nBCs))
    weightmx_CA1_BC[C_CA1_BC.i[:], C_CA1_BC.j[:]] = C_CA1_BC.w_exc[:]

    #return weightmx * 1e9, weightmx_PC_I * 1e9, weightmx_BC_E * 1e9,  weightmx_BC_I * 1e9 # *1e9 nS conversion
    return weightmx, weightmx_PC_I , weightmx_BC_E ,  weightmx_BC_I, weightmx_CA1_PC, weightmx_CA1_BC  

if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[2]
        LoadMatrix = sys.argv[3]
        sim_duration = float(sys.argv[4]) if len(sys.argv) > 4 else 400
    except:
        STDP_mode = "sym"
        LoadMatrix = None
        sim_duration = 400
    assert STDP_mode in ["asym", "sym"]

    place_cell_ratio = 0.5
    linear = True
    f_in = "spike_trains_%.1f_linear.npz" % place_cell_ratio if linear else "spike_trains_%.1f.npz" % place_cell_ratio
    f_in_CA1 = "spike_trains_%.1f_linear_CA1.npz" % place_cell_ratio if linear else "spike_trains_%.1f_CA1.npz" % place_cell_ratio
    f_out = "wmx_%s_%.1f_linear.npz" % (STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)
    #f_in = "intermediate_spike_trains_%.1f_linear.npz" % place_cell_ratio if linear else "intermediate_spike_trains_%.1f.npz" % place_cell_ratio
    #f_out = "intermediate_wmx_%s_%.1f_linear.npz" % (STDP_mode, place_cell_ratio) if linear else "intermediate_wmx_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)

    # STDP parameters (see `optimization/analyse_STDP.py`)
    if STDP_mode == "asym":
        taup = taum = 20 * ms
        #Ap = 0.01
        #Am = -Ap
        #wmax = 4e-8  # S
        #scale_factor = 1.27
    elif STDP_mode == "sym":
        taup = taum = 62.5 * ms
        #Ap = Am = 4e-3
        #wmax = 2e-8  # S
        #scale_factor = 0.62
    #w_init = 1e-10  # S
    ap_ap_range = (0.01, 0.03)  # Range for random Ap selection, in nS. Set to a narrow range to ensure reproducibility while allowing for some variability in the results.
    #Ap = Am = 0.02
    Ap = random.uniform(*ap_ap_range)  # Randomly select Ap from the specified range
    Am = Ap  # Ensure Am is the negative of Ap for symmetry
    wmax_range = (4.0, 4.3)  # Range for random wmax selection, in nS. Set to a narrow range to ensure reproducibility while allowing for some variability in the results.
    #wmax = 4.0 # 
    wmax = random.uniform(*wmax_range)    
    w_init = 0.1
    Ap *= wmax; Am *= wmax  # needed to reproduce Brian1 results

    spiking_neurons, spike_times = load_spike_trains(os.path.join(base_path, "files", f_in))
    try:
        spiking_neurons_CA1, spike_times_CA1 = load_spike_trains(os.path.join(base_path, "files", f_in_CA1))
    except FileNotFoundError:
        print("No separate CA1 spike train found at %s, reusing the CA3/PC spike train for CA1" % f_in_CA1)
        spiking_neurons_CA1, spike_times_CA1 = spiking_neurons, spike_times

    weightmx, wmatrix_PCI, wmatrix_BC_E, wmatrix_BC_I, weightmx_CA1_PC, weightmx_CA1_BC = learning(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, w_init,
                                                                                                     spiking_neurons_CA1=spiking_neurons_CA1, spike_times_CA1=spike_times_CA1, sim_duration=sim_duration)
    #weightmx *= scale_factor  # quick and dirty additional scaling! (in an ideal world the STDP parameters should be changed to include this scaling...)
    #wmatrix_PCI *= scale_factor
    #wmatrix_BC_E *= scale_factor
    #wmatrix_BC_I *= scale_factor

    save_wmx(weightmx, os.path.join(base_path, "files", f_out))
    save_wmx(wmatrix_PCI, os.path.join(base_path, "files", f_out[:-4] + "_PC_I.npz"))
    save_wmx(wmatrix_BC_E, os.path.join(base_path, "files", f_out[:-4] + "_BC_E.npz"))
    save_wmx(wmatrix_BC_I, os.path.join(base_path, "files", f_out[:-4] + "_BC_I.npz"))
    save_wmx(weightmx_CA1_PC, os.path.join(base_path, "files", f_out[:-4] + "_CA1_PC.npz"))
    save_wmx(weightmx_CA1_BC, os.path.join(base_path, "files", f_out[:-4] + "_CA1_BC.npz"))

    #plot_wmx(weightmx, save_name=f_out[:-4])
    #plot_wmx_avg(weightmx, n_pops=100, save_name="%s_avg" % f_out[:-4])
    #plot_w_distr(weightmx, save_name="%s_distr" % f_out[:-4])
    #selection = np.array([500, 2400, 4000, 5500, 7015])
    #plot_weights(save_selected_w(weightmx, selection), save_name="%s_sel_weights" % f_out[:-4])
    device.delete()
    plt.show()
