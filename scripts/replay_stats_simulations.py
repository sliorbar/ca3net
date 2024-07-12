# -*- coding: utf8 -*-
"""
Creates AdExpIF PC and BC populations in Brian2, loads in recurrent connection matrix for PC population
runs simulation and checks the dynamics
authors: András Ecker, Bence Bagi, Szabolcs Káli last update: 07.2019
"""

import os
import sys
import shutil
from brian2.units.allunits import *
from brian2.units.stdunits import *
from brian2.utils.caching import *
import numpy as np
import scipy
import datalayer
import random as pyrandom
import sqlalchemy as sql
import pandas as pd
from brian2 import *
from sqlalchemy import false
from sympy import true
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt
#import brian2genn
from helper import load_wmx, preprocess_monitors, generate_cue_spikes,\
                   save_vars, save_PSD, save_TFR, save_LFP, save_replay_analysis,save_wmx,save_vars_syn,SynWeightDist,save_vars_syn_cpp
from detect_replay import replay_circular, slice_high_activity, replay_linear
from detect_oscillations import analyse_rate, ripple_AC, ripple, gamma, calc_TFR, analyse_estimated_LFP
from plots import plot_violin, plot_raster, plot_posterior_trajectory, plot_PSD, plot_TFR, plot_zoomed, plot_detailed, plot_LFP, set_fig_dir, plot_wmx,set_len_sim,plot_histogram_wmx, plot_Zoom_Weights,fig_dir
#set_device('cpp_standalone', build_on_run=False)
set_device('cpp_standalone', build_on_run=False)

#set_device("cuda_standalone")
#set_device('genn', use_GPU=True, debug=True)
#prefs.devices.genn.connectivity = 'SPARSE'
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
RunType = "org"


##############Start  of LB parameters ###############
org_sim_len = 1000 # First part of the simulation - Can be used to store synaptic weights
first_break_sim_len = 4000 #First break duration in ms can be used to store synaptic weights
end_sim_len = 15000 #Duration in ms of entire simulation
taup_sim = 20 #pre synaptic stdp constant
taum_sim = 20 #post synaptic stdp constant
stdp_post_scale_factor = 0 # Post before pre factor - Positive number is LTD
stdp_pre_scale_factor = 0.1    #Use to modify the pre / post window - Positive number is LTP
total_sim_len=org_sim_len+first_break_sim_len+end_sim_len #Total simulation length
Selected_PC_Index=0 #Index of the selected PC to be used for the detailed synaptic analysis
PC_SynDelay = 2.2 # in ms
Cue_Param = False #True or false for cue
Learning_Rate = 0.01 # Learning rate for STDP (Height of STDP Curve)
synaptic_zoom = 20 # The number of presynaptic connection to log on the zoom PC
adapt_mult = 1 #Adaptation multiplier used for regulating the amount of times PCs spike during replay
cue_start = 1000 #Cue start time in ms
trials = 2 #Number of trials to run
org_run=0 #Run the original simulation
place_cell_ratio = 0.5 #Ratio of place cells to non place cells


##############End of LB parameters ##############
# population size
nPCs = 8000
nBCs = 150
# sparseness
connection_prob_PC = 0.1
connection_prob_BC = 0.25

exp_description = 'Total duration= ' +str(total_sim_len)  + ', synaptic delay = ' +str(PC_SynDelay)+  ', cue = ' +str(Cue_Param)
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
delay_PC_E = PC_SynDelay * ms  # Guzman 2016

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


def run_simulation(wmx_PC_E, STDP_mode, cue, save, save_slice, seed, expdesc = None, engine=None, verbose=True, folder=None, expid=None):
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
    # synaptic weights (see `/optimization/optimize_network.py`)
    w_PC_I = 0.65  # nS
    w_BC_E = 0.85
    w_BC_I = 5.
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
    C_PC_MF.connect(j="i")

    if cue:
        num_of_neurons = 100
        spike_times, spiking_neurons = generate_cue_spikes(neurons=num_of_neurons)
        cue_input = SpikeGeneratorGroup(100, spiking_neurons, spike_times*second)
        # connects at the end of PC pop (...end of track in linear case)
        C_PC_cue = Synapses(cue_input, PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF")
        C_PC_cue.connect(i=np.arange(0, num_of_neurons), j=np.arange(cue_start, cue_start + 100))

    # weight matrix used here
    if STDP_mode == "asym":
        #taup = taum = 20 * ms
        taup = taup_sim * ms 
        taum = taum_sim * ms
        Ap = Learning_Rate
        Am = -Ap * stdp_post_scale_factor # Post syn stdp 
        Ap = Ap * stdp_pre_scale_factor
        #wmax = 2e-8  # S
        scale_factor = 1.27
    elif STDP_mode == "sym":
        taup = taum = 62.5 * ms
        Ap = Am = 4e-3
        wmax = 2e-8  # S
        scale_factor = 0.62
    wmax = np.amax(wmx_PC_E)
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
    
    if RunType == "org":
        C_PC_E = Synapses(PCs, PCs, "w_exc:1", on_pre="x_ampa+=norm_PC_E*w_exc", delay=delay_PC_E)
        C_PC_E.connect(i=wmx_PC_E.row, j=wmx_PC_E.col)
        C_PC_E.w_exc = wmx_PC_E.data
        Selected_PC = C_PC_E.i[Selected_PC_Index]
    
    else:
   
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
        
    synapse_details= exp_description + ', Selected PC=' + str(Selected_PC) + ', Am=' + '{0:.3f}'.format(Am) + ', Ap=' + '{0:.3f}'.format(Ap) + ', taup=' + str(taup) + ', taum=' + str(taum) + ', learning_rate=' + '{0:.3f}'.format(Learning_Rate) + ', adaptation mult={0:.2f}'.format(adapt_mult) + ', cue start=' + str(cue_start)

        

    C_PC_I = Synapses(BCs, PCs, on_pre="x_gaba+=norm_PC_I*w_PC_I", delay=delay_PC_I)
    C_PC_I.connect(p=connection_prob_BC)

    C_BC_E = Synapses(PCs, BCs, on_pre="x_ampa+=norm_BC_E*w_BC_E", delay=delay_BC_E)
    C_BC_E.connect(p=connection_prob_PC)

    C_BC_I = Synapses(BCs, BCs, on_pre="x_gaba+=norm_BC_I*w_BC_I", delay=delay_BC_I)
    C_BC_I.connect(condition="i!=j",p=connection_prob_BC)

    SM_PC = SpikeMonitor(PCs)
    SM_BC = SpikeMonitor(BCs)
    RM_PC = PopulationRateMonitor(PCs)
    RM_BC = PopulationRateMonitor(BCs)

    selection = np.arange(0, nPCs, 20)   # subset of neurons for recoring variables
    StateM_PC = StateMonitor(PCs, variables=["vm", "w", "g_ampa", "g_ampaMF", "g_gaba"], record=selection.tolist(), dt=0.1*ms)
    StateM_BC = StateMonitor(BCs, "vm", record=[nBCs/2], dt=0.1*ms)
    detailed_selection=Selected_PC
    syn_slice = {}
    matrix_df = pd.DataFrame.sparse.from_spmatrix(wmx_PC_E)
    unpivot = matrix_df.melt(ignore_index=False)
    non_zero = unpivot[unpivot.value > 0]
    sortedframe = non_zero.sort_values(by=['value'],ascending=False,inplace=False)
    upstream_neurons = np.array([],dtype=int32)
    if RunType == "org":
        syn_slice = C_PC_E.i[:,detailed_selection]
        PCs_Weights[C_PC_E.i[:], C_PC_E.j[:]] = C_PC_E.w_exc[:]
    else:
        PCs_Weights = matrix_df.to_numpy()
        
    
    
    PCWf_name = os.path.join(folder,'PC_Weights_before')
    PCPf_name = os.path.join(folder,'PC_Weights_Diagram_before')
    '''
    w_exc:1
    '''
    if RunType == "org":
        syn_slice = C_PC_E.i[:,Selected_PC]
        s_w = C_PC_E.w_exc[:,Selected_PC]
    else:
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
    
    datalayer.UpdateTrial(engine=engine,description=expdesc,details=synapse_details,expid=expid)
    
    if RunType == "org":
        net = Network(PCs,BCs,MF,C_PC_MF,C_PC_E,C_PC_I,C_BC_E,C_BC_I, SM_PC,SM_BC,RM_PC,RM_BC,StateM_PC,StateM_BC)
    else:
        C_PC_E_SM = StateMonitor(C_PC_E_STDP,variables =['w_exc'],record=syn_slice,dt=1*ms)
        net = Network(PCs,BCs,MF,C_PC_MF,C_PC_E_STDP,C_PC_I,C_BC_E,C_BC_I, SM_PC,SM_BC,RM_PC,RM_BC,C_PC_E_SM,StateM_PC,StateM_BC)
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
    
    if RunType == "org":
        PCs_Weights_A[C_PC_E.i[:], C_PC_E.j[:]] = C_PC_E.w_exc[:]
    else:
        C_PC_E_STDP_A = C_PC_E_STDP
    if verbose:

        net.run(end_sim_len*ms, report="text")
    else:
        net.run(end_sim_len*ms)
    
    device.build(directory='output', compile=True, run=True, debug=True)
            
    '''
    PCs_Weights_A[C_PC_E_STDP_A.i[:], C_PC_E_STDP_A.j[:]] = C_PC_E_STDP_A.w_exc[:]
    PCWf_name = os.path.join(folder,'PC_Weights_Mid')
    PCPf_name = os.path.join(folder,'PC_Weights_Diagram_Mid')
    plot_wmx(PCs_Weights_A, save_name=PCPf_name)
    plot_histogram_wmx(PCs_Weights_A, save_name=PCPf_name + '_histogram')
    plot_violin(PCs_Weights, PCs_Weights_A, save_name=PCPf_name+'_diff_org_A')
    
    if RunType == "org":
        PCs_Weights_B[C_PC_E.i[:], C_PC_E.j[:]] = C_PC_E.w_exc[:]
    else:
        PCs_Weights_B[C_PC_E_STDP.i[:], C_PC_E_STDP.j[:]] = C_PC_E_STDP.w_exc[:]
    
    PCWf_name = os.path.join(folder,'PC_Weights_end')
    PCPf_name = os.path.join(folder,'PC_Weights_Diagram_end')
    #save_wmx(PCs_Weights, PCWf_name)
    plot_wmx(PCs_Weights_B, save_name=PCPf_name)
    #plot_histogram_wmx(PCs_Weights_B, save_name=PCPf_name + '_histogram')
    plot_violin(PCs_Weights, PCs_Weights_B, save_name=PCPf_name+'_diff_org_B')
    plot_violin(PCs_Weights_A, PCs_Weights_B, save_name=PCPf_name+'_diff_A_B')
    #plot_Zoom_Weights(w=C_PC_E_SM,save_name=PCPf_name+ "_B")
    
    PCs_Weights_Diff = PCs_Weights - PCs_Weights_B
    PCs_Weights_Diff_A = PCs_Weights - PCs_Weights_A
    plot_wmx(PCs_Weights_Diff, save_name=PCPf_name+"_diff")
    plot_wmx(PCs_Weights_Diff_A, save_name=PCPf_name+"_diff_A")   
    df_PCs = SynWeightDist(PCs_Weights)
    df_PCs_A = SynWeightDist(PCs_Weights_A)
    df_PCs_B = SynWeightDist(PCs_Weights_B)
    df_PCs_Diff = SynWeightDist(PCs_Weights_Diff)
    df_PCs_Diff_A = SynWeightDist(PCs_Weights_Diff_A)

    if RunType != "org":
        datalayer.SaveTrial(engine=engine,data=df_PCs,tablename='SynWeightsStats',expid=expid,selected_pc=0)
        datalayer.SaveTrial(engine=engine,data=df_PCs_A,tablename='SynWeightsStats',expid=expid,selected_pc=-1)
        datalayer.SaveTrial(engine=engine,data=df_PCs_B,tablename='SynWeightsStats',expid=expid,selected_pc=-2)
        datalayer.SaveTrial(engine=engine,data=df_PCs_Diff,tablename='SynWeightsStats',expid=expid,selected_pc=-100)
        datalayer.SaveTrial(engine=engine,data=df_PCs_Diff_A,tablename='SynWeightsStats',expid=expid,selected_pc=-99)
    '''
    if save:
        save_vars(SM_PC, RM_PC, StateM_PC, selection, seed)
    if save_slice and RunType != "org" :
        save_vars_syn_cpp(StateM=C_PC_E_SM, folder=fig_dir, SpikeM=SM_PC,SpikeM_BC = SM_BC, selected_pc=detailed_selection, subset = subset_df ,RateM=RM_PC, RateM_BC = RM_BC,engine=engine,expid=expid,offset=0,runType=RunType,synapses=C_PC_E_STDP)
    datalayer.CloseTrial(engine=engine,expid=expid)
    # For iteration with the matrix - Save the synaptic weights
    f_out = "wmx_after_run_%s_%.1f_linear-itr2.npz" % (STDP_mode, place_cell_ratio) if linear else "wmx_after_run_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)
    weightmx = np.zeros((nPCs, nPCs))
    weightmx[C_PC_E_STDP.i[:], C_PC_E_STDP.j[:]] = C_PC_E_STDP.w_exc[:]
    #weightmx =  weightmx * 1e9 #nS conversion
    save_wmx(weightmx, os.path.join(folder, str(expid) +'-wmx_syn_weights_PCs_End.npz'))
    save_wmx(PCs_Weights_Diff, os.path.join(folder, str(expid) +'-wmx_syn_weights_PCs_End_Diff_.npz'))
    #save_wmx(PCs_Weights_A, os.path.join(base_path, "files", 'A_'+f_out))
    return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC



if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]
        STDP_mode_Input = sys.argv[2]
        FolderDescription = sys.argv[3]
        RunT = sys.argv[4]
        Selected_PC_Index = int(sys.argv[5])
    except:
        STDP_mode = "sym"
    assert STDP_mode in ["sym", "asym"]
    assert RunT in ["org", "alt"]
    RunType = RunT
    save = False
    save_slice=True
    cue = Cue_Param
    verbose = True 
    TFR = False
    linear = True
    place_cell_ratio = 0.5
    seed = 12345
    engine = datalayer.InitializeSQLEngine()
    
    expid = datalayer.InitializeTrial(engine=engine,description='temp desc',details='temp detail')
    FolderDescription = str(expid) + '-' + FolderDescription
    
    f_in = "wmx_%s_%.1f_linear.npz"%(STDP_mode_Input, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl" % (STDP_mode_Input, place_cell_ratio)
    PF_pklf_name = os.path.join(base_path, "files", "PFstarts_%s_linear.pkl" % place_cell_ratio) if linear else None
    dir_name = os.path.join(base_path, "figures", "%.2f_replay_det_%s_%.1f" % (1, STDP_mode, place_cell_ratio)) if linear else None
    dir_name_save = os.path.join(base_path, "figures", "%.2f_replay_det_%s_%.1f" % (1, STDP_mode, place_cell_ratio) ,FolderDescription) if linear else None
    set_fig_dir(dir_name_save)
    set_len_sim(total_sim_len)
    if os.path.isdir(dir_name) == False:
        os.mkdir(dir_name)

    if os.path.isdir(dir_name_save) == False:
        os.mkdir(dir_name_save)
        print("dir exist: " + dir_name_save)
    wmx_PC_E = load_wmx(os.path.join(base_path, "files", f_in))     
    engine = datalayer.InitializeSQLEngine()
    SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation(wmx_PC_E, STDP_mode, cue=cue,
                                                                                save=save,save_slice=save_slice,expdesc=FolderDescription, engine=engine, seed=seed, verbose=verbose, folder=dir_name_save,expid=expid)
        
    device.delete()
    plt.show()
  


'''
def parse_arguments(args):
    try:
        STDP_mode = args[0]
        STDP_mode_Input = args[1]
        FolderDescription = args[2]
        RunT = args[3]
        Selected_PC_Index = int(args[4])
    except:
        STDP_mode = "sym"
    assert STDP_mode in ["sym", "asym"]
    assert RunT in ["org", "alt"]
    return STDP_mode, STDP_mode_Input, FolderDescription, RunT, Selected_PC_Index

def initialize_simulation(STDP_mode, STDP_mode_Input, FolderDescription, RunT):
    RunType = RunT
    save = False
    save_slice = True
    cue = Cue_Param
    verbose = True
    TFR = False
    linear = True
    place_cell_ratio = 0.5
    seed = 12345
    engine = datalayer.InitializeSQLEngine()
    expid = datalayer.InitializeTrial(engine=engine, description='temp desc', details='temp detail')
    FolderDescription = f"{expid}-{FolderDescription}"
    f_in = f"wmx_{STDP_mode_Input}_{place_cell_ratio:.1f}_linear.npz" if linear else f"wmx_{STDP_mode_Input}_{place_cell_ratio:.1f}.pkl"
    PF_pklf_name = os.path.join(base_path, "files", f"PFstarts_{place_cell_ratio}_linear.pkl") if linear else None
    dir_name = os.path.join(base_path, "figures", f"{1:.2f}_replay_det_{STDP_mode}_{place_cell_ratio:.1f}") if linear else None
    dir_name_save = os.path.join(dir_name, FolderDescription) if linear else None
    set_fig_dir(dir_name_save)
    set_len_sim(total_sim_len)
    create_directory(dir_name)
    create_directory(dir_name_save)
    return engine, f_in, dir_name_save, cue, save, save_slice, verbose, seed, expid

def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f"Directory created: {path}")

def load_weights(file_path):
    return load_wmx(file_path)

def run_simulation_and_save_results(engine, wmx_PC_E, STDP_mode, cue, save, save_slice, FolderDescription, verbose, dir_name_save, expid, seed):
    SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation(
        wmx_PC_E, STDP_mode, cue=cue, save=save, save_slice=save_slice, expdesc=FolderDescription,
        engine=engine, seed=seed, verbose=verbose, folder=dir_name_save, expid=expid
    )
    return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC

def main(args):
    STDP_mode, STDP_mode_Input, FolderDescription, RunT, Selected_PC_Index = parse_arguments(args)
    engine, f_in, dir_name_save, cue, save, save_slice, verbose, seed, expid = initialize_simulation(
        STDP_mode, STDP_mode_Input, FolderDescription, RunT
    )
    wmx_PC_E = load_weights(os.path.join(base_path, "files", f_in))
    SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation_and_save_results(
        engine, wmx_PC_E, STDP_mode, cue, save, save_slice, FolderDescription, verbose, dir_name_save, expid, seed
    )
    device.delete()
    plt.show()
'''