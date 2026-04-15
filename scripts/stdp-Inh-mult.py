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
import matplotlib.pyplot as plt
import brian2cuda
from helper import load_spike_trains, save_wmx, load_wmx, SynWeightHome, SynWeightHomeUniform
from plots import plot_STDP_rule, plot_wmx, plot_wmx_avg, plot_w_distr, save_selected_w, plot_weights

from brian2 import prefs
#prefs.codegen.target = "numpy"
# Force C++17 for NVCC and for the host compiler it invokes
'''
prefs.codegen.cpp.extra_compile_args = ['-std=c++17']
prefs.codegen.cpp.extra_link_args = []
#prefs.devices.cuda_standalone.profile_kernels = True
#prefs.devices.cuda_standalone.profile_statemonitor_copy_to_host = True
prefs.devices.cuda_standalone.calc_occupancy = True

# more blocks in flight
prefs.devices.cuda_standalone.SM_multiplier = 2  # try 1,2,4

# atomics: can be bottleneck or win depending on contention
prefs.devices.cuda_standalone.use_atomics = True 
# Keep your NVCC-side C++17 too (fine to keep)
prefs.devices.cuda_standalone.cuda_backend.extra_compile_args_nvcc = [
    "-w", "-use_fast_math",
    "--std=c++17",
]
prefs.devices.cpp_standalone.extra_make_args_unix = ["-j8"]
prefs.devices.cpp_standalone.extra_make_args_unix = [
    "CXXFLAGS=-std=c++17",
]
#set_device("cuda_standalone", directory='output_online_sim')
'''
set_device("cpp_standalone",directory='output_online_sim')  # speed up the simulation with generated C++ code


warnings.filterwarnings("ignore")
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
adapt_mult = 1.0
nPCs = 8000
nBCs = 150
#nBCs = 600
plasticity_scale_factor = 0.5  # scaling factor for the STDP window 
# sparseness
connection_prob_PC = 0.1
connection_prob_BC_E = 0.1
connection_prob_BC = 0.25
connection_prob_Conx = 0.05
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

z = 1 * nS
# AdExpIF parameters for PCs (re-optimized by Szabolcs)
g_leak_PC = 4.31475791937223 * nS
tau_mem_PC = 41.7488927175169 * ms
Cm_PC = tau_mem_PC * g_leak_PC
Vrest_PC = -75.1884554193901 * mV
Vreset_PC = -29.738747396665072 * mV
theta_PC = -24.4255910105977 * mV
tref_PC = 5.96326930945599 * ms
delta_T_PC = 4.2340696257631 * mV
spike_th_PC = theta_PC + 5 * delta_T_PC
a_PC = -0.274347065652738 * nS
b_PC = 206.841448096415 * pA
#a_PC = 0 * nS
#b_PC = 0 * pA
#Increasing adaptation by 50%
a_PC *= adapt_mult
b_PC *= adapt_mult
tau_w_PC = 84.9358017225512 * ms
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
#z = 1 * nS
#wmax = 2e-8  # S
# synaptic reversal potentials
Erev_E = 0.0 * mV
Erev_I = -70.0 * mV

eqs_PC = """
dvm/dt = (-g_leak_PC*(vm-Vrest_PC) + g_leak_PC*delta_T_PC*exp((vm- theta_PC)/delta_T_PC) - w - ((g_ampa+g_ampaMF)*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_PC : volt (unless refractory)
dw/dt = (a_PC*(vm-Vrest_PC) - w) / tau_w_PC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_PC_E : 1
dx_ampa/dt = -x_ampa / decay_PC_E : 1
dg_ampaMF/dt = (x_ampaMF - g_ampaMF) / rise_PC_MF : 1
dx_ampaMF/dt = -x_ampaMF / decay_PC_MF : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_PC_I : 1
dx_gaba/dt = -x_gaba/decay_PC_I : 1
"""

tau_w_BC = 178.581099914024 * ms
eqs_BC = """
dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm- theta_BC)/delta_T_BC) - w - (g_ampa*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_BC : volt (unless refractory)
dw/dt = (a_BC*(vm-Vrest_BC) - w) / tau_w_BC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_BC_E : 1
dx_ampa/dt = -x_ampa/decay_BC_E : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_BC_I : 1
dx_gaba/dt = -x_gaba/decay_BC_I : 1
"""


def learning(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, w_init, fin = None, LoadMatrix = None, select_Conx = 1):
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
    max_mult = 3.0  # Allow for maximum 1.5x scaling of the Ph2 weight
    max_mult_BC_E = 3.0  # Allow for maximum 1.5x scaling of the Ph2 weight
    initial_mult = 1.0  # Initial scaling of the weight - 50%
    step_size = 0.01
    inh_tau = 40 * ms
    w_PC_I_inp = 0.65 #* 1e-9 # nS # Taken from Ecker 2022
    w_BC_E_inp = 0.85 #* 1e-9 # nS # Taken from Ecker 2022
    w_BC_I_inp = 5.0 #* 1e-9 # nS # Taken from Ecker 2022
    #wmax_PC_I = 1.0 # Max weight for PC to BC synapses
    #wmax_BC_E = 1.5 # Max weight for BC to PC synapses
    #wmax_BC_I = 8.0 # Max weight for BC to BC synapses
    #wmax_PC_I = w_PC_I_inp * max_mult # Allow for maximum 1.5x scaling of the weight
    #wmax_BC_E = w_BC_E_inp * max_mult_BC_E # Allow for maximum 1.5x scaling of the weight
    #wmax_BC_I = w_BC_I_inp * max_mult # Allow for maximum 1.5x scaling of the weight
    wmax_PC_I = 1.8 # Allow for maximum  scaling of the weight
    wmax_BC_E = 2.4 # Allow for maximum  scaling of the weight
    wmax_BC_I = 10.0 # Allow for maximum scaling of the weight
    w_PC_I_inp = w_PC_I_inp * initial_mult
    w_BC_E_inp = w_BC_E_inp * initial_mult
    w_BC_I_inp = w_BC_I_inp * initial_mult
    
    #PC = SpikeGeneratorGroup(nPCs, spiking_neurons, spike_times*second)
    #sPC = SpikeGeneratorGroup(nPCs, spiking_neurons, spike_times*second) # Spiking PCs based on the generated spike trains, used for learning
    # mimics Brian1's exponential STPD class, with interactions='all', update='additive'
    # see more on conversion: http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html
    #PCs = NeuronGroup(nPCs, model=eqs_PC, threshold="vm>spike_th_PC", reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    #PCs.vm = Vrest_PC; PCs.g_ampa = 0.0; PCs.g_ampaMF = 0.0; PCs.g_gaba = 0.0    
    PCs = SpikeGeneratorGroup(nPCs, spiking_neurons, spike_times*second)
    
    # Conx population - Used to provide Context
    nConx = 8 # Number of context cells
    rate_Conx = 7.0 * Hz #Conx provides context here. at theta frequency, to ensure that they can provide context during the entire simulation, even if they are not perfectly phase-locked to the theta rhythm.
        
    Conx = PoissonGroup(nConx, rate_Conx)
    w_Conx_E = 5.0 # Very strong connection - e.g.
    
    
    
    # Inhinitory population
    BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    BCs.vm  = Vrest_BC; BCs.g_ampa = 0.0; BCs.g_gaba = 0.0    
    #PC to BC plasticity parameters (Ap > 0 is hSTDP)
    Ap_PC_I = -step_size
    Am_PC_I = -Ap_PC_I
    # BC_E plasticity parameters (Ap > 0 is hSTDP)
    Ap_BC_E = -step_size 
    Am_BC_E = -Ap_BC_E
    # BC_I plasticity parameters 
    Ap_BC_I =  step_size 
    Am_BC_I =  Ap_BC_I
    # Scale the plasticity parameters to match the weight range


    Ap_PC_I = wmax_PC_I * Ap_PC_I
    Am_PC_I = wmax_PC_I * Am_PC_I
    Ap_BC_E = wmax_BC_E * Ap_BC_E
    Am_BC_E = wmax_BC_E * Am_BC_E
    Ap_BC_I = wmax_BC_I * Ap_BC_I
    Am_BC_I = wmax_BC_I * Am_BC_I
    dApresyn = Ap
    dApostsyn = Am
    dApresyn_BC_I = Ap_BC_I
    dApostsyn_BC_I = Am_BC_I
    dApresyn_BC_E = Ap_BC_E 
    dApostsyn_BC_E = Am_BC_E
    dApresyn_PC_I = Ap_PC_I
    dApostsyn_PC_I = Am_PC_I
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
    if LoadMatrix == "Y":
        wmx_PC_I = load_wmx(fin[:-4] + "_PC_I.npz")
        wmx_BC_E = load_wmx(fin[:-4] + "_BC_E.npz")
        wmx_BC_I = load_wmx(fin[:-4] + "_BC_I.npz")
        wmx_PC_E = load_wmx(fin)
        #wmx_PC_E.data = SynWeightHome(wmx_PC_E.data, pr_value=wmax*0.8 , top_value=1.0, reset_value=w_init)  # Extract data and apply hoemostasis - Reset the weights to w_init if they are above wmax*0.8, to prevent too high initial weights when loading from file
        #wmx_PC_E.data =  wmx_PC_E.data / scale_factor  # Reverse scaling to get back the original weights
    
    delay_PC_E = 2.2 * ms  # PC to PC synaptic delay, taken from Bartos 2002 (data from DG)
    STDP = Synapses(PCs, PCs,
            """
            w_exc : 1
            dA_presyn/dt = -A_presyn/taup : 1 (event-driven)
            dA_postsyn/dt = -A_postsyn/taum : 1 (event-driven)
            """,
            on_pre="""
            A_presyn += Ap
            w_exc = clip(w_exc + A_postsyn, 0, wmax)
            """,
            on_post="""
            A_postsyn += Am
            w_exc = clip(w_exc + A_presyn, 0, wmax)
            """)
    #C_PC_E_STDP = Synapses(PCs, PCs,synapse_setup, on_pre=on_pre_setup,on_post=on_post_setup,delay=delay_PC_E)
    
    if LoadMatrix == "Y":
        STDP.connect(i=wmx_PC_E.row, j=wmx_PC_E.col)
        STDP.w_exc = wmx_PC_E.data 
    else:
        STDP.connect(condition="i!=j", p=connection_prob_PC)
        STDP.w_exc = w_init
   
    C_PC_I = Synapses(source=PCs, target=BCs, model=synapse_model_PC_I, on_pre= on_pre_setup_PC_I, on_post= on_post_setup_PC_I, delay=delay_PC_I)
    if LoadMatrix == "Y":
        C_PC_I.connect(i=wmx_PC_I.row, j=wmx_PC_I.col)
        C_PC_I.w_e_inh = wmx_PC_I.data
    else:
        C_PC_I.connect(p=connection_prob_BC)
        C_PC_I.w_e_inh = w_PC_I_inp
    # BC to PC 
    C_BC_E = Synapses(source=BCs, target=PCs,model=synapse_model_BC_E, on_pre=on_pre_setup_BC_E, on_post=on_post_setup_BC_E , delay=delay_BC_E)
    
    if LoadMatrix == "Y":
        C_BC_E.connect(i=wmx_BC_E.row, j=wmx_BC_E.col)
        C_BC_E.w_i_exc = wmx_BC_E.data
    else:
        C_BC_E.connect(p=connection_prob_BC_E)
        C_BC_E.w_i_exc = w_BC_E_inp
    # BC to BC
    C_BC_I = Synapses(source=BCs, target=BCs,model=synapse_model_BC_I , on_pre=on_pre_setup_BC_I, on_post= on_post_setup_BC_I, delay=delay_BC_I)
    
    if LoadMatrix == "Y":
        C_BC_I.connect(i=wmx_BC_I.row, j=wmx_BC_I.col)
        C_BC_I.w_i_inh = wmx_BC_I.data
    else:
        C_BC_I.connect(p=connection_prob_BC)
        C_BC_I.w_i_inh = w_BC_I_inp

    # Conx synapses
    w_PC_MF = 100.0 # Very strong connection - e.g. from EC to CA3, as in Guzman 2016
    #sPC_Syn = Synapses(sPC, PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF")
    #sPC_Syn.connect(j="i")  # one-to-one connection from the spike generator to the PCs
    #sPC_Syn.w_Conx_E = w_PC_MF
    
    Conx_Syn = Synapses(Conx, BCs, on_pre="x_ampa+=norm_PC_I*w_Conx_E")
    #Conx_Syn = Synapses(Conx, BCs, on_pre="x_gaba+=norm_BC_I*w_Conx_E")
    if select_Conx == 1:
        Conx_Syn.connect(j="i")
    else:
        Conx_Syn.connect(j="i+nConx")

    ### New context population on PCs
    Conx_Syn_PC = Synapses(Conx, PCs,
            """
            w_exc : 1
            dA_presyn/dt = -A_presyn/taup : 1 (event-driven)
            dA_postsyn/dt = -A_postsyn/taum : 1 (event-driven)
            """,
            on_pre="""
            A_presyn += Ap
            w_exc = clip(w_exc + A_postsyn, 0, wmax)
            """,
            on_post="""
            A_postsyn += Am
            w_exc = clip(w_exc + A_presyn, 0, wmax)
            """)
    Conx_Syn_PC.connect(p=connection_prob_Conx)  # Sparse connection from Conx to PCs
    Conx_Syn_PC.w_exc = w_init  # Scale the weight by the number of Conx neurons to keep total inital input constant
    #device.build(directory='output_online_sim', compile=True, run=False, debug=True, clean=True)
    # Increased from 20s to match original stdp.py approach
    # For fast testing use 60-100s, for full learning use 200-400s
    SM_PC = SpikeMonitor(PCs)
    SM_BC = SpikeMonitor(BCs)
    #SM_sPC = SpikeMonitor(PCs)
    run(400*second, report="text")
    print("Total spikes - SM_PC:", SM_PC.num_spikes)
    print("Total spikes - SM_BC:", SM_BC.num_spikes)
    #print("Total spikes - SM_sPC:", SM_sPC.num_spikes)
    weightmx = np.zeros((nPCs, nPCs))
    weightmx[STDP.i[:], STDP.j[:]] = STDP.w_exc[:]

    weightmx_PC_I = np.zeros((nPCs, nBCs))
    weightmx_PC_I[C_PC_I.i[:], C_PC_I.j[:]] = C_PC_I.w_e_inh[:]
    weightmx_BC_E = np.zeros((nBCs, nPCs))
    weightmx_BC_E[C_BC_E.i[:], C_BC_E.j[:]] = C_BC_E.w_i_exc[:]
    weightmx_BC_I = np.zeros((nBCs, nBCs))
    weightmx_BC_I[C_BC_I.i[:], C_BC_I.j[:]] = C_BC_I.w_i_inh[:]
    weightmx_Conx_PC = np.zeros((nConx, nPCs))
    weightmx_Conx_PC[Conx_Syn_PC.i[:], Conx_Syn_PC.j[:]] = Conx_Syn_PC.w_exc[:]

    #return weightmx * 1e9, weightmx_PC_I * 1e9, weightmx_BC_E * 1e9,  weightmx_BC_I * 1e9 # *1e9 nS conversion
    return weightmx, weightmx_PC_I , weightmx_BC_E ,  weightmx_BC_I , weightmx_Conx_PC  # *1e9 nS conversion

if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[2]
        LoadMatrix = sys.argv[3]
        select_Conx = int(sys.argv[4])
    except:
        STDP_mode = "sym"
        LoadMatrix = "N"
        select_Conx = 1
    assert STDP_mode in ["asym", "sym"]

    place_cell_ratio = 0.5
    linear = True
    f_in = "spike_trains_%.1f_linear.npz" % place_cell_ratio if linear else "spike_trains_%.1f.npz" % place_cell_ratio
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
        #taup = taum = 20.0 * ms
        #Ap = Am = 4e-3  # Same ratio as original (4e-3 * wmax)
        #wmax = 2e-8  # S
        #scale_factor = 0.62
        #wmax = wmax * scale_factor
    # w_init in same units as wmax (dimensionless, represents nS)
    # Original: 1e-10 S = 0.0001 nS, which is 0.5% of 2e-8 S = 20 nS
    # For wmax=7 nS, 0.5% would be 0.035, but start even smaller
    # w_init = 1e-10  # dimensionless (represents 0.00035 nS, ~0.005% of wmax)
    Ap = Am = 0.02
    wmax = 4.0 # 
    w_init = 0.1
    Ap *= wmax; Am *= wmax  # needed to reproduce Brian1 results

    spiking_neurons, spike_times = load_spike_trains(os.path.join(base_path, "files", f_in))

    weightmx, wmatrix_PCI, wmatrix_BC_E, wmatrix_BC_I, wmatrix_Conx_PC = learning(spiking_neurons = spiking_neurons, spike_times = spike_times, taup = taup, taum = taum, Ap = Ap, Am = Am, wmax = wmax, w_init = w_init, select_Conx = select_Conx, LoadMatrix = LoadMatrix, fin = os.path.join(base_path, "files", f_out))
    #if LoadMatrix != "Y":
    #    weightmx *= scale_factor  # quick and dirty additional scaling! (in an ideal world the STDP parameters should be changed to include this scaling...)
    #print("Applied scale factor: %.2f" % scale_factor)
    #weightmx *= scale_factor
    #wmatrix_PCI *= scale_factor
    #wmatrix_BC_E *= scale_factor
    #wmatrix_BC_I *= scale_factor

    save_wmx(weightmx, os.path.join(base_path, "files", f_out))
    save_wmx(wmatrix_PCI, os.path.join(base_path, "files", f_out[:-4] + "_PC_I.npz"))
    save_wmx(wmatrix_BC_E, os.path.join(base_path, "files", f_out[:-4] + "_BC_E.npz"))
    save_wmx(wmatrix_BC_I, os.path.join(base_path, "files", f_out[:-4] + "_BC_I.npz"))
    save_wmx(wmatrix_Conx_PC, os.path.join(base_path, "files", f_out[:-4] + "_Conx_PC_%.1f.npz" % select_Conx))

    #plot_wmx(weightmx, save_name=f_out[:-4])
    #plot_wmx_avg(weightmx, n_pops=100, save_name="%s_avg" % f_out[:-4])
    #plot_w_distr(weightmx, save_name="%s_distr" % f_out[:-4])
    #selection = np.array([500, 2400, 4000, 5500, 7015])
    #plot_weights(save_selected_w(weightmx, selection), save_name="%s_sel_weights" % f_out[:-4])
    device.delete()
    #plt.show()
