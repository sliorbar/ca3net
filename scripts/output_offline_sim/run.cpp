#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include<random>

#include "code_objects/neurongroup_1_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_1_spike_resetter_codeobject_1.h"
#include "code_objects/neurongroup_1_spike_resetter_codeobject_2.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject_1.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject_2.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject_1.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject_2.h"
#include "code_objects/neurongroup_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_spike_resetter_codeobject_1.h"
#include "code_objects/neurongroup_spike_resetter_codeobject_2.h"
#include "code_objects/neurongroup_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_spike_thresholder_codeobject_1.h"
#include "code_objects/neurongroup_spike_thresholder_codeobject_2.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject_1.h"
#include "code_objects/neurongroup_stateupdater_codeobject_2.h"
#include "code_objects/poissongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/poissongroup_1_spike_thresholder_codeobject_1.h"
#include "code_objects/poissongroup_1_spike_thresholder_codeobject_2.h"
#include "code_objects/poissongroup_spike_thresholder_codeobject.h"
#include "code_objects/poissongroup_spike_thresholder_codeobject_1.h"
#include "code_objects/poissongroup_spike_thresholder_codeobject_2.h"
#include "code_objects/ratemonitor_1_codeobject.h"
#include "code_objects/ratemonitor_1_codeobject_1.h"
#include "code_objects/ratemonitor_1_codeobject_2.h"
#include "code_objects/ratemonitor_codeobject.h"
#include "code_objects/ratemonitor_codeobject_1.h"
#include "code_objects/ratemonitor_codeobject_2.h"
#include "code_objects/spikemonitor_1_codeobject.h"
#include "code_objects/spikemonitor_1_codeobject_1.h"
#include "code_objects/spikemonitor_1_codeobject_2.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/spikemonitor_codeobject_1.h"
#include "code_objects/spikemonitor_codeobject_2.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/statemonitor_codeobject_1.h"
#include "code_objects/statemonitor_codeobject_2.h"
#include "code_objects/synapses_1_post_codeobject.h"
#include "code_objects/synapses_1_post_codeobject_1.h"
#include "code_objects/synapses_1_post_codeobject_2.h"
#include "code_objects/synapses_1_post_push_spikes.h"
#include "code_objects/synapses_1_pre_codeobject.h"
#include "code_objects/synapses_1_pre_codeobject_1.h"
#include "code_objects/synapses_1_pre_codeobject_2.h"
#include "code_objects/synapses_1_pre_push_spikes.h"
#include "code_objects/synapses_1_synapses_create_array_codeobject.h"
#include "code_objects/synapses_2_post_codeobject.h"
#include "code_objects/synapses_2_post_codeobject_1.h"
#include "code_objects/synapses_2_post_codeobject_2.h"
#include "code_objects/synapses_2_post_push_spikes.h"
#include "code_objects/synapses_2_pre_codeobject.h"
#include "code_objects/synapses_2_pre_codeobject_1.h"
#include "code_objects/synapses_2_pre_codeobject_2.h"
#include "code_objects/synapses_2_pre_push_spikes.h"
#include "code_objects/synapses_2_synapses_create_array_codeobject.h"
#include "code_objects/synapses_3_post_codeobject.h"
#include "code_objects/synapses_3_post_codeobject_1.h"
#include "code_objects/synapses_3_post_codeobject_2.h"
#include "code_objects/synapses_3_post_push_spikes.h"
#include "code_objects/synapses_3_pre_codeobject.h"
#include "code_objects/synapses_3_pre_codeobject_1.h"
#include "code_objects/synapses_3_pre_codeobject_2.h"
#include "code_objects/synapses_3_pre_push_spikes.h"
#include "code_objects/synapses_3_synapses_create_array_codeobject.h"
#include "code_objects/synapses_4_post_codeobject.h"
#include "code_objects/synapses_4_post_codeobject_1.h"
#include "code_objects/synapses_4_post_codeobject_2.h"
#include "code_objects/synapses_4_post_push_spikes.h"
#include "code_objects/synapses_4_pre_codeobject.h"
#include "code_objects/synapses_4_pre_codeobject_1.h"
#include "code_objects/synapses_4_pre_codeobject_2.h"
#include "code_objects/synapses_4_pre_push_spikes.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject.h"
#include "code_objects/synapses_5_pre_codeobject.h"
#include "code_objects/synapses_5_pre_codeobject_1.h"
#include "code_objects/synapses_5_pre_codeobject_2.h"
#include "code_objects/synapses_5_pre_push_spikes.h"
#include "code_objects/synapses_5_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_6_post_codeobject.h"
#include "code_objects/synapses_6_post_codeobject_1.h"
#include "code_objects/synapses_6_post_codeobject_2.h"
#include "code_objects/synapses_6_post_push_spikes.h"
#include "code_objects/synapses_6_pre_codeobject.h"
#include "code_objects/synapses_6_pre_codeobject_1.h"
#include "code_objects/synapses_6_pre_codeobject_2.h"
#include "code_objects/synapses_6_pre_push_spikes.h"
#include "code_objects/synapses_6_synapses_create_array_codeobject.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_pre_codeobject_1.h"
#include "code_objects/synapses_pre_codeobject_2.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/synapses_synapses_create_generator_codeobject.h"


void brian_start()
{
	_init_arrays();
	_load_arrays();
	// Initialize clocks (link timestep and dt to the respective arrays)
    brian::defaultclock.timestep = brian::_array_defaultclock_timestep;
    brian::defaultclock.t = brian::_array_defaultclock_t;
    brian::defaultclock.dt = brian::_array_defaultclock_dt;
    brian::statemonitor_clock.timestep = brian::_array_statemonitor_clock_timestep;
    brian::statemonitor_clock.t = brian::_array_statemonitor_clock_t;
    brian::statemonitor_clock.dt = brian::_array_statemonitor_clock_dt;
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}


