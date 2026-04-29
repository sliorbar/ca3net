# -*- coding: utf8 -*-
"""
Creates AdExpIF PC and BC populations in Brian2, loads in recurrent connection matrix for PC population
runs simulation and checks the dynamics
authors: András Ecker, Bence Bagi, Szabolcs Káli last update: 07.2019
"""

import os
import secrets
import sys
import shutil
from brian2.units.allunits import *
from brian2.units.stdunits import *
from brian2.utils.caching import *
import numpy as np
import scipy
import datalayerOmen
import random as pyrandom
import sqlalchemy as sql
import pandas as pd
from brian2 import *
from sqlalchemy import false
from sympy import true
#prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt
#import brian2cuda
import random
from scipy.sparse import coo_matrix
#import brian2genn
from helper import load_wmx, preprocess_monitors, generate_cue_spikes,\
                   save_vars, save_PSD, save_TFR, save_LFP, save_replay_analysis,save_wmx,save_vars_syn,SynWeightDist,save_vars_syn_cpp, SynWeightHome, SynWeightHomeUniform, _load_PF_starts
from detect_replay import replay_circular, slice_high_activity, replay_linear
from detect_oscillations import analyse_rate, ripple_AC, ripple, gamma, calc_TFR, analyse_estimated_LFP
from plots import plot_violin, plot_raster, plot_posterior_trajectory, plot_PSD, plot_TFR, plot_zoomed, plot_detailed, plot_LFP, set_fig_dir, plot_wmx,set_len_sim,plot_histogram_wmx, plot_Zoom_Weights,fig_dir
from brian2 import prefs

# Force C++17 for NVCC and for the host compiler it invokes
#prefs.codegen.cpp.extra_compile_args = ['-std=c++17']
#prefs.codegen.cpp.extra_link_args = []

# Keep your NVCC-side C++17 too (fine to keep)
#prefs.devices.cuda_standalone.cuda_backend.extra_compile_args_nvcc = [
#    "-w", "-use_fast_math",
#    "--std=c++17",
#]
#prefs.devices.cpp_standalone.extra_make_args_unix = ["-j8"]
#prefs.devices.cpp_standalone.extra_make_args_unix = [
#    "CXXFLAGS=-std=c++17",
#]
set_device('cpp_standalone', build_on_run=False)
#set_device('cuda_standalone', build_on_run=False)


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
#RunType = "org"


##############Start  of LB parameters ###############
org_sim_len = 1000 # First part of the simulation - Can be used to store synaptic weights
first_break_sim_len = 4000 #First break duration in ms can be used to store synaptic weights
end_sim_len = 10000 #Duration in ms of entire simulation
#taup_sim = 20 #pre synaptic stdp constant
#taum_sim = 20 #post synaptic stdp constant
#stdp_post_scale_factor = -0.1 # Post before pre factor - Positive number is LTD
#stdp_pre_scale_factor = -0.1    #Use to modify the pre / post window - Positive number is LTP
total_sim_len=org_sim_len+first_break_sim_len+end_sim_len #Total simulation length
Selected_PC_Index=0 #Index of the selected PC to be used for the detailed synaptic analysis
PC_SynDelay = 2.2 # in ms
Cue_Param = False #True or false for cue
#Learning_Rate = 0.01 # Learning rate for STDP (Height of STDP Curve)
synaptic_zoom = 20 # The number of presynaptic connection to log on the zoom PC
adapt_mult = 1 #Adaptation multiplier used for regulating the amount of times PCs spike during replay
cue_start = 1000 #Cue start location PC index (used only if Cue_Param is True)
#trials = 2 #Number of trials to run
#org_run=0 #Run the original simulation
#place_cell_ratio = 0.3 #Ratio of place cells to non place cells


##############End of LB parameters ##############
# population size
BC_mult = 1.0 # Multiplier for the number of BCs - Used to test the effect of increasing the number of BCs in the network
nPCs = 8000
#nBCs = 150
nBCs = 300 
# sparseness
#connection_prob_PC = 0.1
#connection_prob_BC = 0.25

exp_description = 'Total duration= ' +str(total_sim_len) +  ', cue = ' +str(Cue_Param)
# synaptic time constants:
# rise time constants
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
# synaptic delays:
#delay_PC_E = 2.2 * ms  # Guzman 2016
delay_PC_I = 1.1 * ms  # Bartos 2002
delay_BC_E = 0.9 * ms  # Geiger 1997 (data from DG)
delay_BC_I = 0.6 * ms  # Bartos 2002
'''Modify the code to remove synaptic delay'''
#delay_PC_E = PC_SynDelay * ms  # Guzman 2016

# synaptic reversal potentials
Erev_E = 0.0 * mV
Erev_I = -70.0 * mV

rate_MF = 15.0 * Hz  # mossy fiber input freq

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
""" comment this back to run with ExpIF PC model...
# ExpIF parameters for PCs (optimized by Szabolcs)
g_leak_PC = 4.88880734814042 * nS
tau_mem_PC = 70.403501012992 * ms
Cm_PC = tau_mem_PC * g_leak_PC
Vrest_PC = -76.59966923496779 * mV
Vreset_PC = -58.8210432444992 * mV
theta_PC = -28.7739788756 * mV
tref_PC = 1.07004414539699 * ms
delta_T_PC = 10.7807538634886 * mV
spike_th_PC = theta_PC + 5 * delta_T_PC
a_PC = 0. * nS
b_PC = 0. * pA
tau_w_PC = 1 * ms
"""
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

eqs_BC = """
dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm- theta_BC)/delta_T_BC) - w - (g_ampa*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_BC : volt (unless refractory)
dw/dt = (a_BC*(vm-Vrest_BC) - w) / tau_w_BC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_BC_E : 1
dx_ampa/dt = -x_ampa/decay_BC_E : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_BC_I : 1
dx_gaba/dt = -x_gaba/decay_BC_I : 1
"""


#def run_simulation(wmx_PC_E, STDP_mode, cue, save, save_slice, seed, expdesc = None, engine=None, verbose=True, folder=None, expid=None):
def run_simulation(wmx_PC_E,wmx_PC_I, wmx_BC_E, wmx_BC_I, wmx_Conx_PC, STDP_mode, cue, save, save_slice, seed, expdesc=None, engine=None, verbose=True, folder=None, expid=None,
                   taup_sim=20, taum_sim=20, stdp_post_scale_factor=-0.1, stdp_pre_scale_factor=-0.1, delay_PC_E=2.2, Learning_Rate=0.01,connection_prob_PC = 0.1, connection_prob_BC = 0.25, place_cell_ratio=0.5, select_Conx = 1, connection_prob_BC_E=0.25, STDP_mode_Input = "sym", syn_preserve = 1.0, PF_pklf_name = None, tau_inh = 20, stdp_inh_scale_factor = 0.1, inh_max_weight = 2.0, end_duration_length = 10000, wmax = 4.0, do_not_save = "N"):

    """
    Sets up the network and runs simulation
    :param wmx_PC_E: np.array representing the recurrent excitatory synaptic weight matrix
    :param STDP_mode: asym/sym STDP mode used for the learning (see `stdp.py`) - here used only to set the weights
    :param cue: if True it adds an other Brian2 `SpikeGeneratorGroup` to stimulate a subpop in the beginning (cued replay)
    :param save: bool flag to save PC spikes after the simulation (used by `bayesian_decoding.py` later)
    :param seed: random seed used for running the simulation
    :param verbose: bool flag to report status of simulation
    :return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC: Brian2 monitors (+ array of selected cells used by multi state monitor)
    """
    global org_run
    np.random.seed(seed)
    pyrandom.seed(seed)
    global Selected_PC_Index
    inh_plasticity_training = True  # If True, inhibitory plasticity is enabled during the training phase
    inh_plasticity = True
    #max_inhibition_mult = 1.5  # Maximum scaling of inhibitory weights
    max_inhibition_mult_PC_I = 1.0  # Maximum scaling of inhibitory weights for PC to BC synapses
    max_inhibition_mult_BC_E = 1.0  # Maximum scaling of inhibitory weights for BC to PC synapses
    max_inhibition_mult_BC_I = 1.0 # Maximum scaling of inhibitory weights for BC to BC synapses
    max_excitation_mult_PC_E = 1.0  # Maximum scaling of excitatory weights for PC to PC synapses
    step_size = stdp_inh_scale_factor # Step size for inhibitory plasticity updates - Used to calculate Ap and Am for inhibitory plasticity
    # synaptic weights (see `/optimization/optimize_network.py`)
    w_PC_I_input = 0.65  # nS
    w_BC_E_input = 0.85  # nS
    w_BC_I_input = 5.  # nS
    w_init = 0.01 # Initial weight for synapses with homeostasis (used in SynWeightHome function)
    if STDP_mode == "asym":
        w_PC_MF = 21.5
    elif STDP_mode == "sym":
        w_PC_MF = 19.15
    else:
        raise ValueError("STDP_mode has to be either 'sym' or 'asym'!")

    PCs = NeuronGroup(nPCs, model=eqs_PC, threshold="vm>spike_th_PC",
                      reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    PCs.vm = Vrest_PC; PCs.g_ampa = 0.0; PCs.g_ampaMF = 0.0; PCs.g_gaba = 0.0

    BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    BCs.vm  = Vrest_BC; BCs.g_ampa = 0.0; BCs.g_gaba = 0.0

    MF = PoissonGroup(nPCs, rate_MF)
    C_PC_MF = Synapses(MF, PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF")
    
    #pf_Starts = _load_PF_starts(PF_pklf_name)
    #PCdata = pd.DataFrame({'PC_Index': list(pf_Starts.keys())})
    #MF_block = np.arange(0, nPCs)
    #MF_targets = np.intersect1d(MF_block, PCdata["PC_Index"].to_numpy(dtype=int))
    #num_of_MF_eurons = len(MF_targets)
    #C_PC_MF.connect(i=np.arange(0, num_of_MF_eurons), j=MF_targets)
    C_PC_MF.connect(j="i")

    # Conx population - Used to provide Context
    nConx = 1 # Number of context cells
    rate_Conx = 0.0001 * Hz #Conx provides context here. 
    
    Conx = PoissonGroup(nConx, rate_Conx)
    w_Conx_E = 5.0 # Connection weight for context to BC synapses
    # Save the place cells in the database
    pf_Starts = _load_PF_starts(PF_pklf_name)
    PCdata = pd.DataFrame({
        'PC_Index': list(pf_Starts.keys()),
        'PF_Start': list(pf_Starts.values()),
    })
    PCdata = PCdata.sort_values('PF_Start', kind='mergesort').reset_index(drop=True)
    PCdata['PC_Order'] = PCdata.index.astype(int)
    PCdata_for_save = PCdata[['PC_Index', 'PC_Order']].copy()
    datalayerOmen.SaveTrial(engine=engine, expid=expid, data=PCdata_for_save, tablename="place_field_selected")
    if cue:
        cue_block_size = 100
        cue_block = np.arange(cue_start, cue_start + cue_block_size)
        cue_orders = np.intersect1d(cue_block, PCdata_for_save["PC_Order"].to_numpy(dtype=int))
        cue_targets = PCdata_for_save.loc[
            PCdata_for_save["PC_Order"].isin(cue_orders), "PC_Index"
        ].to_numpy(dtype=int)
        num_of_neurons = len(cue_targets)
        spike_times, spiking_neurons = generate_cue_spikes(neurons=num_of_neurons)
        cue_input = SpikeGeneratorGroup(num_of_neurons, spiking_neurons, spike_times*second)
        # connects at the end of PC pop (...end of track in linear case)
        C_PC_cue = Synapses(cue_input, PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF")
        C_PC_cue.connect(i=np.arange(0, num_of_neurons), j=cue_targets)

    # weight matrix used here
    if STDP_mode == "asym":
        #taup = taum = 20 * ms
        taup = taup_sim * ms 
        taum = taum_sim * ms
        Ap = Learning_Rate
        Am = Ap * stdp_post_scale_factor # Post syn stdp 
        Ap = Ap * stdp_pre_scale_factor
        #wmax = 2e-8  # S
        scale_factor = 1.27
        print("Using asymmetric STDP")
    elif STDP_mode == "sym":
        taup = taum = 62.5 * ms
        Ap = Am = 4e-3
        #wmax = 2e-8  # S
        scale_factor = 0.62
        print("Using symmetric STDP")
    #wmax = np.amax(wmx_PC_E) * max_excitation_mult_PC_E  # Allow for maximum scaling of the PC to PC weight
    #wmax = 4.0 # in nS
    Ap *= wmax
    Am *= wmax 
    #To align with code in Brian2 documentation (https://brian2.readthedocs.io/en/latest/examples/frompapers.Izhikevich_2007.html?highlight=stdp#example-izhikevich-2007)
    dApresyn = Ap
    dApostsyn = Am 

    synapse_setup='''
    w_exc:1
    dApresyn/dt = -Apresyn/taup : 1 (event-driven)
    dApostsyn/dt = -Apostsyn/taum : 1 (event-driven)
    '''
    on_pre_setup = '''
    x_ampa+=norm_PC_E*w_exc
    Apresyn += dApresyn
    w_exc = clip(w_exc + Apostsyn,0,wmax)
    '''
    on_post_setup= '''
    Apostsyn += dApostsyn
    w_exc = clip(w_exc + Apresyn,0,wmax)
    '''
          
    PCs_Weights = np.zeros((nPCs, nPCs))
    PCs_Weights_A = np.zeros((nPCs, nPCs))
    PCs_Weights_B = np.zeros((nPCs, nPCs))
    PCs_Weights_Diff = np.zeros((nPCs, nPCs))
    total_sim_len= org_sim_len + first_break_sim_len + end_duration_length
    exp_description = 'Total duration= ' +str(total_sim_len) +  ', cue = ' +str(Cue_Param)
    synapse_details= exp_description + ', synaptic delay = {0:.2f}'.format(delay_PC_E)+ ', Am=' + '{0:.3f}'.format(Am) + ', Ap=' + '{0:.3f}'.format(Ap) + ', taup=' + '{0:.3f}'.format(taup) + ', taum=' + '{0:.3f}'.format(taum) + ', learning_rate=' + '{0:.3f}'.format(Learning_Rate) + ', adaptation mult={0:.2f}'.format(adapt_mult) + ', cue start=' + str(cue_start) + ' , STDP mode=' + STDP_mode + ', connection_prob_PC=' + '{0:.2f}'.format(connection_prob_PC) + ', connection_prob_BC=' + '{0:.2f}'.format(connection_prob_BC)
    

    delay_PC_E = delay_PC_E * ms
     
    C_PC_E_STDP = Synapses(PCs, PCs,synapse_setup, on_pre=on_pre_setup,on_post=on_post_setup,delay=delay_PC_E)
    C_PC_E_STDP.connect(i=wmx_PC_E.row, j=wmx_PC_E.col)
    C_PC_E_STDP.w_exc= wmx_PC_E.data
    if Selected_PC_Index < 0:
        pc_sel = C_PC_E_STDP.i[:]
        rng = np.random.default_rng()
        Selected_PC = rng.choice(a=pc_sel)
    else:
        Selected_PC = wmx_PC_E.row[Selected_PC_Index]
        print(Selected_PC)
    

# Synapse plasticity rules for BCs
    # Max weights for inhibitory plasticity   
    wmax_PC_I = 0.65* inh_max_weight
    wmax_BC_E = 0.85 * inh_max_weight
    wmax_BC_I = 5.0 * (inh_max_weight * 0.5)
    #wmax_PC_I = 2.0 # Allow for maximum scaling of the weight
    #wmax_BC_E = 2.4 # Allow for maximum scaling of the weight
    #wmax_BC_I = 10.0 # Allow for maximum scaling of the weight
        
    if inh_plasticity == True:
        Ap_PC_I = -step_size
        Am_PC_I = Ap_PC_I * -1.0
    else:
        Ap_PC_I = 0.0
        Am_PC_I = 0.0
    # BC_E plasticity parameters (Ap > 0 is hSTDP)
    if inh_plasticity == True:
        Ap_BC_E = -step_size
        Am_BC_E = Ap_BC_E * -1.0
    else:
        Ap_BC_E = 0.0
        Am_BC_E = 0.0
    # BC_I plasticity parameters (Ap > 0 is hSTDP)
    if inh_plasticity == True:
        Ap_BC_I = - step_size
        Am_BC_I = Ap_BC_I * -1.0 ## This is for symmetric inhibitory plasticity on BC to BC synapses
        #Ap_BC_I = step_size
        #Am_BC_I = Ap_BC_I  ## This is for symmetric inhibitory plasticity on BC to BC synapses
    else:
        Ap_BC_I = 0.0
        Am_BC_I = 0.0
    # Time constants for inhibitory plasticity
    if inh_plasticity == True:
        tau_PC_I = tau_inh * ms
        tau_BC_I = tau_inh * ms
        tau_BC_E = tau_inh * ms
    else:
        tau_PC_I = 1.0 * ms
        tau_BC_I = 1.0 * ms
        tau_BC_E = 1.0 * ms

    Ap_PC_I = wmax_PC_I * Ap_PC_I 
    Am_PC_I = wmax_PC_I * Am_PC_I 
    Ap_BC_E = wmax_BC_E * Ap_BC_E 
    Am_BC_E = wmax_BC_E * Am_BC_E 
    Ap_BC_I = wmax_BC_I * Ap_BC_I 
    Am_BC_I = wmax_BC_I * Am_BC_I 
    synapse_details = synapse_details + ', Ap_BC_I=' + '{0:.3f}'.format(Ap_BC_I) + ', Am_BC_I=' + '{0:.3f}'.format(Am_BC_I) + ', Ap_PC_I=' + '{0:.3f}'.format(Ap_PC_I) + ', Am_PC_I=' + '{0:.3f}'.format(Am_PC_I) + ', Ap_BC_E=' + '{0:.3f}'.format(Ap_BC_E) + ', Am_BC_E=' + '{0:.3f}'.format(Am_BC_E)
    synapse_details = synapse_details + ', Tau_BC_I=' + '{0:.3f}'.format(tau_BC_I) + ', Tau_BC_E=' + '{0:.3f}'.format(tau_BC_E) + ', Tau_PC_I=' + '{0:.3f}'.format(tau_PC_I) + ', wmax_PC_I=' + '{0:.3f}'.format(wmax_PC_I) + ', wmax_BC_E=' + '{0:.3f}'.format(wmax_BC_E) + ', wmax_BC_I=' + '{0:.3f}'.format(wmax_BC_I) + ', inh_plasticity=' + str(inh_plasticity) + ', wmax=' + '{0:.2f}'.format(wmax) + ', synaptic_preserve=' + '{0:.2f}'.format(syn_preserve) + ', end_duration=' + str(end_duration_length)
    print(synapse_details)
    #dApresyn = Ap
    #dApostsyn = Am
    dApresyn_BC_I = Ap_BC_I
    dApostsyn_BC_I = Am_BC_I
    dApresyn_BC_E = Ap_BC_E 
    dApostsyn_BC_E = Am_BC_E
    dApresyn_PC_I = Ap_PC_I
    dApostsyn_PC_I = Am_PC_I
    #tau_bc = 12 * ms  # Different tau for BCs
    # PC_I modeling
    synapse_model_PC_I='''
    w_PC_I:1
    dApresyn_PC_I/dt = -Apresyn_PC_I/tau_PC_I : 1 (event-driven)
    dApostsyn_PC_I/dt = -Apostsyn_PC_I/tau_PC_I : 1 (event-driven)
    '''
    on_pre_setup_PC_I = '''
    x_ampa+=norm_PC_I*w_PC_I
    Apresyn_PC_I += dApresyn_PC_I
    w_PC_I = clip(w_PC_I + Apostsyn_PC_I,0,wmax_PC_I)
    '''
    on_post_setup_PC_I= '''
    Apostsyn_PC_I += dApostsyn_PC_I
    w_PC_I = clip(w_PC_I + Apresyn_PC_I,0,wmax_PC_I)
    '''
    # BC_E modeling
    synapse_model_BC_E='''
    w_BC_E:1
    dApresyn_BC_E/dt = -Apresyn_BC_E/tau_BC_E : 1 (event-driven)
    dApostsyn_BC_E/dt = -Apostsyn_BC_E/tau_BC_E : 1 (event-driven)
    '''
    on_pre_setup_BC_E = '''
    x_gaba+=norm_BC_E*w_BC_E
    Apresyn_BC_E += dApresyn_BC_E
    w_BC_E = clip(w_BC_E + Apostsyn_BC_E,0,wmax_BC_E)
    '''
    on_post_setup_BC_E= '''
    Apostsyn_BC_E += dApostsyn_BC_E
    w_BC_E = clip(w_BC_E + Apresyn_BC_E,0,wmax_BC_E)
    '''
    # BC_I modeling
    synapse_model_BC_I='''
    w_BC_I:1
    dApresyn_BC_I/dt = -Apresyn_BC_I/tau_BC_I : 1 (event-driven)
    dApostsyn_BC_I/dt = -Apostsyn_BC_I/tau_BC_I : 1 (event-driven)
    '''
    on_pre_setup_BC_I = '''
    x_gaba+=norm_BC_I*w_BC_I
    Apresyn_BC_I += dApresyn_BC_I
    w_BC_I = clip(w_BC_I + Apostsyn_BC_I,0,wmax_BC_I)
    '''
    on_post_setup_BC_I= '''
    Apostsyn_BC_I += dApostsyn_BC_I
    w_BC_I = clip(w_BC_I + Apresyn_BC_I,0,wmax_BC_I)
    '''
    # Synapses definition
    

    C_PC_I = Synapses(source=PCs, target=BCs, model=synapse_model_PC_I, on_pre= on_pre_setup_PC_I, on_post= on_post_setup_PC_I, delay=delay_PC_I)
    #C_PC_I = Synapses(PCs, BCs, model="w_PC_I:1", on_pre="x_ampa+=norm_PC_I*w_PC_I", delay=delay_PC_I)
    if inh_plasticity_training == True:
        C_PC_I.connect(i=wmx_PC_I.row, j=wmx_PC_I.col)
        C_PC_I.w_PC_I = wmx_PC_I.data
    else:
        C_PC_I.connect(p=connection_prob_BC)
        C_PC_I.w_PC_I = w_PC_I_input
    
    C_BC_E = Synapses(source=BCs, target=PCs,model=synapse_model_BC_E, on_pre=on_pre_setup_BC_E, on_post=on_post_setup_BC_E , delay=delay_BC_E)
    #C_BC_E = Synapses(BCs, PCs, model="w_BC_E:1", on_pre="x_gaba+=norm_BC_E*w_BC_E", delay=delay_BC_E)
    if inh_plasticity_training == True:
        C_BC_E.connect(i=wmx_BC_E.row, j=wmx_BC_E.col)
        C_BC_E.w_BC_E = wmx_BC_E.data
    else:
        C_BC_E.connect(p=connection_prob_PC)
        C_BC_E.w_BC_E = w_BC_E_input
    

    C_BC_I = Synapses(source=BCs, target=BCs,model=synapse_model_BC_I , on_pre=on_pre_setup_BC_I, on_post= on_post_setup_BC_I, delay=delay_BC_I)
    #C_BC_I = Synapses(BCs, BCs, model="w_BC_I:1", on_pre="x_gaba+=norm_BC_I*w_BC_I", delay=delay_BC_I)
    if  inh_plasticity_training == True:
        C_BC_I.connect(i=wmx_BC_I.row, j=wmx_BC_I.col)
        C_BC_I.w_BC_I = wmx_BC_I.data
    else:
        C_BC_I.connect(condition="i!=j",p=connection_prob_BC_E)
        C_BC_I.w_BC_I = w_BC_I_input

    #Conx synapses
    w_Conx_E = 0.0 # Set the initial weight for context synapses to 0.0 nS
    Conx_Syn = Synapses(Conx, BCs, on_pre="x_ampa+=norm_PC_I*w_Conx_E")
    #Conx_Syn = Synapses(Conx, BCs, on_pre="x_gaba+=norm_BC_I*w_Conx_E")
    if select_Conx == 1:
        Conx_Syn.connect(j="i")
    else:
        Conx_Syn.connect(j="i+nConx")
    
    Conx_Syn_PC = Synapses(Conx, PCs,synapse_setup, on_pre=on_pre_setup,on_post=on_post_setup,delay=delay_PC_E)
    Conx_Syn_PC.connect(i=wmx_Conx_PC.row, j=wmx_Conx_PC.col)
    Conx_Syn_PC.w_exc= wmx_Conx_PC.data
    
    #Run the simulation
    SM_PC = SpikeMonitor(PCs)
    SM_BC = SpikeMonitor(BCs)
    RM_PC = PopulationRateMonitor(PCs)
    RM_BC = PopulationRateMonitor(BCs)

    selection = np.arange(0, nPCs, 20)   # subset of neurons for recoring variables
    #StateM_PC = StateMonitor(PCs, variables=["vm", "w", "g_ampa", "g_ampaMF", "g_gaba"], record=selection.tolist(), dt=0.1*ms)
    #StateM_BC = StateMonitor(BCs, "vm", record=[nBCs/2], dt=0.1*ms)
    detailed_selection=Selected_PC
    syn_slice = {}
    matrix_df = pd.DataFrame.sparse.from_spmatrix(wmx_PC_E)
    unpivot = matrix_df.melt(ignore_index=False)
    non_zero = unpivot[unpivot.value > 0]
    sortedframe = non_zero.sort_values(by=['value'],ascending=False,inplace=False)
    upstream_neurons = np.array([],dtype=int32)
    PCs_Weights = matrix_df.to_numpy()
        
    
    
    PCWf_name = os.path.join(folder,'PC_Weights_before')
    PCPf_name = os.path.join(folder,'PC_Weights_Diagram_before')
    '''
    w_exc:1
    '''
    
    index_syn = matrix_df.melt(ignore_index=False)
    index_syn = index_syn.loc[index_syn["value"] > 0]
    index_syn = index_syn.reset_index()
    index_syn["i"] = index_syn.index
    s_w = index_syn.loc[index_syn["variable"] == Selected_PC]
    
             

    subset_df = pd.DataFrame()
    subset_df["index"]=s_w["i"]
    subset_df["pc"]=s_w["index"]
    subset_df["value"]=s_w["value"]
    subset_df.sort_values(by=['value'],ascending=False,inplace=True)
    subset_df = subset_df.head(synaptic_zoom)
    subset_df.sort_values(by=['index'],ascending=True,inplace=True)
    upstream_neurons = subset_df['index']

    syn_slice = upstream_neurons.values
    
    datalayerOmen.UpdateTrial(engine=engine,description=expdesc,details=synapse_details,expid=expid)
    
    
    C_PC_E_StateM = StateMonitor(C_PC_E_STDP,variables =['w_exc'],record=syn_slice,dt=1*ms)
    #net = Network(PCs,BCs,MF,C_PC_MF,C_PC_E_STDP,C_PC_I,C_BC_E,C_BC_I, SM_PC,SM_BC,RM_PC,RM_BC,C_PC_E_StateM,StateM_PC,StateM_BC, Conx, Conx_Syn, Conx_Syn_PC)
    net = Network(PCs,BCs,MF,C_PC_MF,C_PC_E_STDP,C_PC_I,C_BC_E,C_BC_I, SM_PC,SM_BC,RM_PC,RM_BC,C_PC_E_StateM, Conx, Conx_Syn, Conx_Syn_PC)
    #net = Network(PCs,BCs,MF,C_PC_MF,C_PC_E_STDP,C_PC_I,C_BC_E,C_BC_I, SM_PC,SM_BC,RM_PC,RM_BC,C_PC_E_StateM,StateM_PC,StateM_BC)
    if cue:
        net.add(C_PC_cue,cue_input)
    if verbose:
        net.run(org_sim_len*ms, report="text")
    
    else:
        net.run(org_sim_len*ms)

    if verbose:
        net.run(first_break_sim_len*ms, report="text")
        

    else:
        net.run(first_break_sim_len*ms)
    
    C_PC_E_STDP_A = C_PC_E_STDP
    
    if verbose:

        net.run(end_duration_length*ms, report="text")
    else:
        net.run(end_duration_length*ms)
    
    device.build(directory='output_offline_sim', compile=True, run=True, clean=True)
    
    if save:
        #save_vars(SM_PC, RM_PC, StateM_PC, selection, seed)
        save_vars(SM_PC, RM_PC,  selection, seed)
    if save_slice :
        save_vars_syn_cpp(StateM=C_PC_E_StateM, folder=fig_dir, SpikeM=SM_PC,SpikeM_BC = SM_BC, selected_pc=detailed_selection, subset = subset_df ,RateM=RM_PC, RateM_BC = RM_BC,engine=engine,expid=expid,offset=0,runType="alt",synapses=C_PC_E_STDP, do_not_save=do_not_save)
        
    datalayerOmen.CloseTrial(engine=engine,expid=expid)
    # For iteration with the matrix - Save the synaptic weights
    f_out = "wmx_after_run_%s_%.1f_linear-itr2.npz" % (STDP_mode, place_cell_ratio) if linear else "wmx_after_run_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)
    weightmx = np.zeros((nPCs, nPCs))
    weightmx_PC_I = np.zeros((nPCs, nBCs))
    weightmx_BC_E = np.zeros((nBCs, nPCs))
    weightmx_BC_I = np.zeros((nBCs, nBCs))
    # Set values larger than 1e-10
    min_val = 0
    mask = C_PC_E_STDP.w_exc[:] > min_val # Create a mask for values greater than 1e-10 * 0.62
    weightmx[C_PC_E_STDP.i[:], C_PC_E_STDP.j[:]] = C_PC_E_STDP.w_exc[:]
    max_weightmx = np.amax(weightmx)
    print("Max weight after simulation: {0:.3f}".format(max_weightmx))
    weightmx = SynWeightHome(weightmx, pr_value=syn_preserve, reset_value=w_init)
    weightmx_PC_I[C_PC_I.i[:], C_PC_I.j[:]] = C_PC_I.w_PC_I[:]
    weightmx_BC_E[C_BC_E.i[:], C_BC_E.j[:]] = C_BC_E.w_BC_E[:]
    weightmx_BC_I[C_BC_I.i[:], C_BC_I.j[:]] = C_BC_I.w_BC_I[:]
    f_out_matrix = "wmx_%s_%.1f_linear.npz" % (STDP_mode_Input, place_cell_ratio)
    save_wmx(weightmx, os.path.join(base_path, "files", f_out_matrix))
    save_wmx(weightmx_PC_I, os.path.join(base_path, "files", f_out_matrix[:-4] + "_PC_I.npz"))
    save_wmx(weightmx_BC_E, os.path.join(base_path, "files", f_out_matrix[:-4] + "_BC_E.npz"))
    save_wmx(weightmx_BC_I, os.path.join(base_path, "files", f_out_matrix[:-4] + "_BC_I.npz"))
    
    #weightmx[C_PC_E_STDP.i[mask], C_PC_E_STDP.j[mask]] = C_PC_E_STDP.w_exc[mask]
    #weightmx =  weightmx * 1e9 #nS conversion
    #PCs_Weights_filtered = np.where(PCs_Weights > min_val, PCs_Weights, 0)
    #save_wmx(weightmx, os.path.join(folder, str(expid) +'-wmx_syn_weights_PCs_End.npz'))
    #save_wmx(PCs_Weights, os.path.join(folder, str(expid) +'-wmx_syn_weights_PCs_start.npz'))
    #save_wmx(PCs_Weights_A, os.path.join(base_path, "files", 'A_'+f_out))
    #return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC, weightmx
    return SM_PC, SM_BC, RM_PC, RM_BC, selection,  weightmx





def remap_weight_matrix_blocks(weightmx):
    """
    Remaps neuron indices in a COO sparse weight matrix by swapping blocks:
    1000–1999 <--> 4000–4999.

    :param weightmx: scipy.sparse.coo_matrix, shape (n, n)
    :return: remapped scipy.sparse.coo_matrix with updated row/col indices
    """
    assert isinstance(weightmx, coo_matrix), "Input must be a COO sparse matrix"
    n = weightmx.shape[0]
    assert weightmx.shape[0] == weightmx.shape[1], "Matrix must be square"
    assert n >= 5000, "Matrix must have at least 5000 rows/columns"

    # Build identity map, then swap the two blocks
    idx_map = np.arange(n)
    idx_map[1000:2000], idx_map[4000:5000] = (
        idx_map[4000:5000].copy(),
        idx_map[1000:2000].copy()
    )

    # Apply remapping to the row and column indices
    new_rows = idx_map[weightmx.row]
    new_cols = idx_map[weightmx.col]

    # Reconstruct new COO matrix
    return coo_matrix((weightmx.data, (new_rows, new_cols)), shape=weightmx.shape)




if __name__ == "__main__":
    try:
        STDP_mode = sys.argv[1]
        STDP_mode_Input = sys.argv[2]
        FolderDescription = sys.argv[3]
        CueT = sys.argv[4]
        Selected_PC_Index = int(sys.argv[5])
        syn_preserve = float(sys.argv[6])
        select_Conx = int(sys.argv[7])
        PF_pklf_name_postfix = sys.argv[8]
        end_duration_length = int(sys.argv[9]) if len(sys.argv) > 9 else 10000
        save_PC_weights = sys.argv[10] if len(sys.argv) > 10 else None
        do_not_save = sys.argv[11] if len(sys.argv) > 11 else "N"
    except:
        STDP_mode = "sym"
        select_Conx = 1
        syn_preserve = 1.0
        PF_pklf_name_postfix = None
        end_duration_length = 10000
        save_PC_weights = None
        do_not_save = "N"
    assert STDP_mode in ["sym", "asym"]
    assert CueT in ["N", "Y"]
    #RunType = RunT
    save = False
    save_slice = True
    if CueT == "Y":
        Cue_Param = True
    else:   
        Cue_Param = False
    cue = Cue_Param
    verbose = True 
    TFR = False
    linear = True
    seed = secrets.randbits(31)
    #seed = 12345

    # Set ranges for each parameter
    taup_sim_range = (10.0, 15.0)  # Example range for taup_sim
    taum_sim_range = (15, 20)  # Example range for taum_sim
    stdp_pre_scale_factor_range = (-0.02, -0.01)  # Example range for stdp_pre_scale_factor
    stdp_post_scale_factor_range = (0.01, 0.02)  # Example range for stdp_post_scale_factor
    PC_SynDelay_range = (2.2, 2.3)  # Example range for PC_SynDelay
    Learning_Rate_range = (0.01, 0.02)  # Example range for Learning_Rate
    place_cell_ratio_range = (0.3, 0.3)  # Example range for place_cell_ratio
    connection_prob_PC_range = (0.1,0.1)
    connection_prob_BC_range = (0.25,0.25)
    syn_preserve_range = (2.5,4.0) # Range for synaptic preservation for synaptic tagging homeostasis in nS
    tau_inh_range = (15, 20) # Range for inhibitory plasticity time constant in ms
    stdp_inh_scale_factor_range = (0.01, 0.02) # Range for inhibitory plasticity scale factor (Ap and Am will be calculated based on this and the max weight)
    inh_max_weight_range = (1.5, 2.0) # Range multiplier for maximum weight for inhibitory synapses in nS
    wmax_range = (3.5, 4.0) # Range for maximum weight for excitatory synapses in nS
    # Initialize SQL engine
    engine = datalayerOmen.InitializeSQLEngine()

    # Randomly select parameters from the defined ranges
    taup_sim = random.uniform(*taup_sim_range)
    #taum_sim = random.uniform(*taum_sim_range)
    #stdp_post_scale_factor = random.uniform(*stdp_post_scale_factor_range)
    stdp_pre_scale_factor = random.uniform(*stdp_pre_scale_factor_range) 
    
    # Make stdp kernel asymmetric
    stdp_post_scale_factor = stdp_pre_scale_factor *-1 #Asymetric STDP
    #stdp_post_scale_factor = stdp_pre_scale_factor # symmetric STDP
    taum_sim = taup_sim
    
    PC_SynDelay = random.uniform(*PC_SynDelay_range)
    Learning_Rate = random.uniform(*Learning_Rate_range)
    tau_inh = random.uniform(*tau_inh_range)
    stdp_inh_scale_factor = random.uniform(*stdp_inh_scale_factor_range)
    inh_max_weight = random.uniform(*inh_max_weight_range)
    wmax = random.uniform(*wmax_range) 
    #syn_preserve = random.uniform(*syn_preserve_range)
    #place_cell_ratio = random.uniform(*place_cell_ratio_range)
    #connection_prob_PC = random.uniform(*connection_prob_PC_range)
    #connection_prob_BC = random.uniform(*connection_prob_BC_range)
    connection_prob_PC = 0.1
    connection_prob_BC = 0.25 
    connection_prob_BC_E = 0.1 
    place_cell_ratio = 0.5
    
    # Update folder description for each combination
    expid = datalayerOmen.InitializeTrial(engine=engine, description='syn-compression', 
                                        details=f'taup_sim={taup_sim}, taum_sim={taum_sim}, PC_SynDelay={PC_SynDelay}')
    FolderDescription = f"{expid}-{FolderDescription}-MC_taup_{taup_sim:.2f}_Ap-scape_{stdp_pre_scale_factor:.2f}_delay_{PC_SynDelay:.2f}"
    
    # Set input and output file names based on current parameters
    f_in = f"wmx_{STDP_mode_Input}_{place_cell_ratio:.1f}_linear.npz" if linear else f"wmx_{STDP_mode_Input}_{place_cell_ratio:.1f}.pkl"
    f_in_PC_I = f"wmx_{STDP_mode_Input}_{place_cell_ratio:.1f}_linear_PC_I.npz" if linear else f"wmx_{STDP_mode_Input}_{place_cell_ratio:.1f}_PC_I.pkl"
    f_in_BC_E = f"wmx_{STDP_mode_Input}_{place_cell_ratio:.1f}_linear_BC_E.npz" if linear else f"wmx_{STDP_mode_Input}_{place_cell_ratio:.1f}_BC_E.pkl"
    f_in_BC_I = f"wmx_{STDP_mode_Input}_{place_cell_ratio:.1f}_linear_BC_I.npz" if linear else f"wmx_{STDP_mode_Input}_{place_cell_ratio:.1f}_BC_I.pkl"
    f_in_Conx_PC = f"wmx_{STDP_mode_Input}_{place_cell_ratio:.1f}_linear_Conx_PC_{select_Conx:.1f}.npz" if linear else f"wmx_{STDP_mode_Input}_{place_cell_ratio:.1f}_Conx_PC_{select_Conx:.1f}.pkl"
    if PF_pklf_name_postfix is not None:
        PF_pklf_name = os.path.join(base_path, "files", f"PFstarts_{place_cell_ratio}_linear_{PF_pklf_name_postfix}.pkl") if linear else None
    else:
        PF_pklf_name = os.path.join(base_path, "files", f"PFstarts_{place_cell_ratio}_linear.pkl") if linear else None
    dir_name = os.path.join(base_path, "figures", f"{1:.2f}_replay_det_{STDP_mode}_{place_cell_ratio:.1f}") if linear else None
    dir_name_save = os.path.join(base_path, "figures", f"{1:.2f}_replay_det_{STDP_mode}_{place_cell_ratio:.1f}", FolderDescription) if linear else None
    
    # Ensure directories exist
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    if not os.path.isdir(dir_name_save):
        os.mkdir(dir_name_save)
        print("Created dir: " + dir_name_save)
    
    # Load weight matrix
    wmx_PC_E = load_wmx(os.path.join(base_path, "files", f_in)) #Weight matrix for PC_E
    #wmx_PC_E.data = SynWeightHome(wmx_PC_E.data,pr_value=syn_preserve) # Reset low weights below threshold
    wmx_PC_I = load_wmx(os.path.join(base_path, "files", f_in_PC_I)) #Weight matrix for PC_I
    wmx_BC_E = load_wmx(os.path.join(base_path, "files", f_in_BC_E)) #Weight matrix for BC_E
    wmx_BC_I = load_wmx(os.path.join(base_path, "files", f_in_BC_I)) #Weight matrix for BC_I
    wmx_Conx_PC = load_wmx(os.path.join(base_path, "files", f_in_Conx_PC)) #Weight matrix for Conx to PC synapses
    #wmx_PC_E = remap_weight_matrix_blocks(wmx_PC_E)  # Remap the weight matrix blocks if needed
    wmax = np.max(wmx_PC_E)
    # Run simulation with the randomly selected parameters
    #SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC, weightmx = run_simulation(
    #    wmx_PC_E=wmx_PC_E, wmx_PC_I=wmx_PC_I, wmx_BC_E=wmx_BC_E, wmx_BC_I=wmx_BC_I, STDP_mode=STDP_mode, cue=cue, save=save, save_slice=save_slice, expdesc=FolderDescription,
    #    engine=engine, seed=seed, verbose=verbose, folder=dir_name_save, expid=expid,
    #    taup_sim=taup_sim, taum_sim=taum_sim, stdp_post_scale_factor=stdp_post_scale_factor, 
    #    stdp_pre_scale_factor=stdp_pre_scale_factor, delay_PC_E=PC_SynDelay, Learning_Rate=Learning_Rate,connection_prob_PC=connection_prob_PC,connection_prob_BC=connection_prob_BC, select_Conx=select_Conx, connection_prob_BC_E = connection_prob_BC_E, STDP_mode_Input=STDP_mode_Input)
    #SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC, weightmx = run_simulation(
    SM_PC, SM_BC, RM_PC, RM_BC, selection, weightmx = run_simulation(
        wmx_PC_E=wmx_PC_E, wmx_PC_I=wmx_PC_I, wmx_BC_E=wmx_BC_E, wmx_BC_I=wmx_BC_I, wmx_Conx_PC=wmx_Conx_PC, STDP_mode=STDP_mode, cue=cue, save=save, save_slice=save_slice, expdesc=FolderDescription,
        engine=engine, seed=seed, verbose=verbose, folder=dir_name_save, expid=expid, PF_pklf_name = PF_pklf_name,
        taup_sim=taup_sim, taum_sim=taum_sim, stdp_post_scale_factor=stdp_post_scale_factor, 
        stdp_pre_scale_factor=stdp_pre_scale_factor, delay_PC_E=PC_SynDelay, Learning_Rate=Learning_Rate,connection_prob_PC=connection_prob_PC,connection_prob_BC=connection_prob_BC, STDP_mode_Input = STDP_mode_Input, connection_prob_BC_E = connection_prob_BC_E, select_Conx=select_Conx, syn_preserve=syn_preserve, inh_max_weight = inh_max_weight, tau_inh = tau_inh, stdp_inh_scale_factor = stdp_inh_scale_factor, end_duration_length=end_duration_length, wmax=wmax, do_not_save=do_not_save)
    
    output_w = SynWeightHome(weightmx,syn_preserve) 
    flattened_weights = output_w.flatten()
    weight_counts = pd.Series({
        "syn_preserve_value": float(syn_preserve),
        "below_syn_preserve": int(np.sum((flattened_weights <= syn_preserve) & (flattened_weights > 0 ))),
        "above_syn_preserve": int(np.sum(flattened_weights > syn_preserve)),
    })
    print("Synaptic weight summary after homeostatic compression:")
    print(weight_counts)
    datalayerOmen.SaveTrial(engine=engine, expid=expid,tablename="synaptic_weights_bins",data=[weight_counts.to_dict()])
    nonzero_weights = flattened_weights[flattened_weights > 0]
    bin_size = 0.5
    max_w = float(np.ceil(nonzero_weights.max() / bin_size) * bin_size) if len(nonzero_weights) > 0 else bin_size
    bin_edges = np.arange(0.0, max_w + bin_size, bin_size)
    counts, _ = np.histogram(nonzero_weights, bins=bin_edges)
    weight_bin_rows = [
        {"bin_low": float(bin_edges[i]), "bin_high": float(bin_edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]
    datalayerOmen.SaveTrial(engine=engine, expid=expid, tablename="synaptic_weights_histogram", data=weight_bin_rows)
    if save_PC_weights == "Y":
        save_wmx(weightmx, os.path.join(base_path, "files", f_in))
    

    device.delete()
    #plt.show()

  
