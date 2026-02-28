#include <stdlib.h>
#include "objects.h"
#include <csignal>
#include <ctime>
#include <time.h>

#include "run.h"
#include "brianlib/common_math.h"

#include "code_objects/neurongroup_1_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_1_spike_resetter_codeobject_1.h"
#include "code_objects/neurongroup_1_spike_resetter_codeobject_2.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject_1.h"
#include "code_objects/after_run_neurongroup_1_spike_thresholder_codeobject_1.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject_2.h"
#include "code_objects/after_run_neurongroup_1_spike_thresholder_codeobject_2.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject_1.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject_2.h"
#include "code_objects/neurongroup_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_spike_resetter_codeobject_1.h"
#include "code_objects/neurongroup_spike_resetter_codeobject_2.h"
#include "code_objects/neurongroup_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_spike_thresholder_codeobject_1.h"
#include "code_objects/after_run_neurongroup_spike_thresholder_codeobject_1.h"
#include "code_objects/neurongroup_spike_thresholder_codeobject_2.h"
#include "code_objects/after_run_neurongroup_spike_thresholder_codeobject_2.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject_1.h"
#include "code_objects/neurongroup_stateupdater_codeobject_2.h"
#include "code_objects/poissongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/after_run_poissongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/poissongroup_1_spike_thresholder_codeobject_1.h"
#include "code_objects/after_run_poissongroup_1_spike_thresholder_codeobject_1.h"
#include "code_objects/poissongroup_1_spike_thresholder_codeobject_2.h"
#include "code_objects/after_run_poissongroup_1_spike_thresholder_codeobject_2.h"
#include "code_objects/poissongroup_spike_thresholder_codeobject.h"
#include "code_objects/after_run_poissongroup_spike_thresholder_codeobject.h"
#include "code_objects/poissongroup_spike_thresholder_codeobject_1.h"
#include "code_objects/after_run_poissongroup_spike_thresholder_codeobject_1.h"
#include "code_objects/poissongroup_spike_thresholder_codeobject_2.h"
#include "code_objects/after_run_poissongroup_spike_thresholder_codeobject_2.h"
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
#include "code_objects/before_run_synapses_1_post_push_spikes.h"
#include "code_objects/before_run_synapses_1_post_push_spikes.h"
#include "code_objects/before_run_synapses_1_post_push_spikes.h"
#include "code_objects/synapses_1_pre_codeobject.h"
#include "code_objects/synapses_1_pre_codeobject_1.h"
#include "code_objects/synapses_1_pre_codeobject_2.h"
#include "code_objects/synapses_1_pre_push_spikes.h"
#include "code_objects/before_run_synapses_1_pre_push_spikes.h"
#include "code_objects/before_run_synapses_1_pre_push_spikes.h"
#include "code_objects/before_run_synapses_1_pre_push_spikes.h"
#include "code_objects/synapses_1_synapses_create_array_codeobject.h"
#include "code_objects/synapses_2_post_codeobject.h"
#include "code_objects/synapses_2_post_codeobject_1.h"
#include "code_objects/synapses_2_post_codeobject_2.h"
#include "code_objects/synapses_2_post_push_spikes.h"
#include "code_objects/before_run_synapses_2_post_push_spikes.h"
#include "code_objects/before_run_synapses_2_post_push_spikes.h"
#include "code_objects/before_run_synapses_2_post_push_spikes.h"
#include "code_objects/synapses_2_pre_codeobject.h"
#include "code_objects/synapses_2_pre_codeobject_1.h"
#include "code_objects/synapses_2_pre_codeobject_2.h"
#include "code_objects/synapses_2_pre_push_spikes.h"
#include "code_objects/before_run_synapses_2_pre_push_spikes.h"
#include "code_objects/before_run_synapses_2_pre_push_spikes.h"
#include "code_objects/before_run_synapses_2_pre_push_spikes.h"
#include "code_objects/synapses_2_synapses_create_array_codeobject.h"
#include "code_objects/synapses_3_post_codeobject.h"
#include "code_objects/synapses_3_post_codeobject_1.h"
#include "code_objects/synapses_3_post_codeobject_2.h"
#include "code_objects/synapses_3_post_push_spikes.h"
#include "code_objects/before_run_synapses_3_post_push_spikes.h"
#include "code_objects/before_run_synapses_3_post_push_spikes.h"
#include "code_objects/before_run_synapses_3_post_push_spikes.h"
#include "code_objects/synapses_3_pre_codeobject.h"
#include "code_objects/synapses_3_pre_codeobject_1.h"
#include "code_objects/synapses_3_pre_codeobject_2.h"
#include "code_objects/synapses_3_pre_push_spikes.h"
#include "code_objects/before_run_synapses_3_pre_push_spikes.h"
#include "code_objects/before_run_synapses_3_pre_push_spikes.h"
#include "code_objects/before_run_synapses_3_pre_push_spikes.h"
#include "code_objects/synapses_3_synapses_create_array_codeobject.h"
#include "code_objects/synapses_4_post_codeobject.h"
#include "code_objects/synapses_4_post_codeobject_1.h"
#include "code_objects/synapses_4_post_codeobject_2.h"
#include "code_objects/synapses_4_post_push_spikes.h"
#include "code_objects/before_run_synapses_4_post_push_spikes.h"
#include "code_objects/before_run_synapses_4_post_push_spikes.h"
#include "code_objects/before_run_synapses_4_post_push_spikes.h"
#include "code_objects/synapses_4_pre_codeobject.h"
#include "code_objects/synapses_4_pre_codeobject_1.h"
#include "code_objects/synapses_4_pre_codeobject_2.h"
#include "code_objects/synapses_4_pre_push_spikes.h"
#include "code_objects/before_run_synapses_4_pre_push_spikes.h"
#include "code_objects/before_run_synapses_4_pre_push_spikes.h"
#include "code_objects/before_run_synapses_4_pre_push_spikes.h"
#include "code_objects/synapses_4_synapses_create_array_codeobject.h"
#include "code_objects/synapses_5_pre_codeobject.h"
#include "code_objects/synapses_5_pre_codeobject_1.h"
#include "code_objects/synapses_5_pre_codeobject_2.h"
#include "code_objects/synapses_5_pre_push_spikes.h"
#include "code_objects/before_run_synapses_5_pre_push_spikes.h"
#include "code_objects/before_run_synapses_5_pre_push_spikes.h"
#include "code_objects/before_run_synapses_5_pre_push_spikes.h"
#include "code_objects/synapses_5_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_6_post_codeobject.h"
#include "code_objects/synapses_6_post_codeobject_1.h"
#include "code_objects/synapses_6_post_codeobject_2.h"
#include "code_objects/synapses_6_post_push_spikes.h"
#include "code_objects/before_run_synapses_6_post_push_spikes.h"
#include "code_objects/before_run_synapses_6_post_push_spikes.h"
#include "code_objects/before_run_synapses_6_post_push_spikes.h"
#include "code_objects/synapses_6_pre_codeobject.h"
#include "code_objects/synapses_6_pre_codeobject_1.h"
#include "code_objects/synapses_6_pre_codeobject_2.h"
#include "code_objects/synapses_6_pre_push_spikes.h"
#include "code_objects/before_run_synapses_6_pre_push_spikes.h"
#include "code_objects/before_run_synapses_6_pre_push_spikes.h"
#include "code_objects/before_run_synapses_6_pre_push_spikes.h"
#include "code_objects/synapses_6_synapses_create_array_codeobject.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_pre_codeobject_1.h"
#include "code_objects/synapses_pre_codeobject_2.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/before_run_synapses_pre_push_spikes.h"
#include "code_objects/before_run_synapses_pre_push_spikes.h"
#include "code_objects/before_run_synapses_pre_push_spikes.h"
#include "code_objects/synapses_synapses_create_generator_codeobject.h"


#include <iostream>
#include <fstream>
#include <string>


        std::string _format_time(float time_in_s)
        {
            float divisors[] = {24*60*60, 60*60, 60, 1};
            char letters[] = {'d', 'h', 'm', 's'};
            float remaining = time_in_s;
            std::string text = "";
            int time_to_represent;
            for (int i =0; i < sizeof(divisors)/sizeof(float); i++)
            {
                time_to_represent = int(remaining / divisors[i]);
                remaining -= time_to_represent * divisors[i];
                if (time_to_represent > 0 || text.length())
                {
                    if(text.length() > 0)
                    {
                        text += " ";
                    }
                    text += (std::to_string(time_to_represent)+letters[i]);
                }
            }
            //less than one second
            if(text.length() == 0)
            {
                text = "< 1s";
            }
            return text;
        }
        void report_progress(const double elapsed, const double completed, const double start, const double duration)
        {
            if (completed == 0.0)
            {
                std::cout << "Starting simulation at t=" << start << " s for duration " << duration << " s";
            } else
            {
                std::cout << completed*duration << " s (" << (int)(completed*100.) << "%) simulated in " << _format_time(elapsed);
                if (completed < 1.0)
                {
                    const int remaining = (int)((1-completed)/completed*elapsed+0.5);
                    std::cout << ", estimated " << _format_time(remaining) << " remaining.";
                }
            }

            std::cout << std::endl << std::flush;
        }
        


void set_from_command_line(const std::vector<std::string> args)
{
    for (const auto& arg : args) {
		// Split into two parts
		size_t equal_sign = arg.find("=");
		auto name = arg.substr(0, equal_sign);
		auto value = arg.substr(equal_sign + 1, arg.length());
		brian::set_variable_by_name(name, value);
	}
}

void _int_handler(int signal_num) {
	if (Network::_globally_running && !Network::_globally_stopped) {
		Network::_globally_stopped = true;
	} else {
		std::signal(signal_num, SIG_DFL);
		std::raise(signal_num);
	}
}

int main(int argc, char **argv)
{
	std::signal(SIGINT, _int_handler);
	std::random_device _rd;
	std::vector<std::string> args(argv + 1, argv + argc);
	if (args.size() >=2 && args[0] == "--results_dir")
	{
		brian::results_dir = args[1];
		#ifdef DEBUG
		std::cout << "Setting results dir to '" << brian::results_dir << "'" << std::endl;
		#endif
		args.erase(args.begin(), args.begin()+2);
	}
        

	brian_start();
        

	{
		using namespace brian;

		
                
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        
                        
                        for(int i=0; i<_num__array_neurongroup_lastspike; i++)
                        {
                            _array_neurongroup_lastspike[i] = - 10000.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_not_refractory; i++)
                        {
                            _array_neurongroup_not_refractory[i] = true;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_vm; i++)
                        {
                            _array_neurongroup_vm[i] = - 0.0751884554193901;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_g_ampa; i++)
                        {
                            _array_neurongroup_g_ampa[i] = 0.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_g_ampaMF; i++)
                        {
                            _array_neurongroup_g_ampaMF[i] = 0.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_g_gaba; i++)
                        {
                            _array_neurongroup_g_gaba[i] = 0.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_lastspike; i++)
                        {
                            _array_neurongroup_1_lastspike[i] = - 10000.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_not_refractory; i++)
                        {
                            _array_neurongroup_1_not_refractory[i] = true;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_vm; i++)
                        {
                            _array_neurongroup_1_vm[i] = - 0.07474167987795019;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_g_ampa; i++)
                        {
                            _array_neurongroup_1_g_ampa[i] = 0.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_g_gaba; i++)
                        {
                            _array_neurongroup_1_g_gaba[i] = 0.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_poissongroup_rates; i++)
                        {
                            _array_poissongroup_rates[i] = 15.0;
                        }
                        
        _run_synapses_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_num__array_poissongroup_1_rates; i++)
                        {
                            _array_poissongroup_1_rates[i] = 14.0;
                        }
                        
        _dynamic_array_synapses_1_delay.resize(1);
        _dynamic_array_synapses_1_delay.resize(1);
        _dynamic_array_synapses_1_delay[0] = 0.002227670402342876;
        
                        
                        for(int i=0; i<_num__array_synapses_1_sources; i++)
                        {
                            _array_synapses_1_sources[i] = _static_array__array_synapses_1_sources[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_synapses_1_targets; i++)
                        {
                            _array_synapses_1_targets[i] = _static_array__array_synapses_1_targets[i];
                        }
                        
        _run_synapses_1_synapses_create_array_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_1_w_exc.size(); i++)
                        {
                            _dynamic_array_synapses_1_w_exc[i] = _static_array__dynamic_array_synapses_1_w_exc[i];
                        }
                        
        _dynamic_array_synapses_2_delay.resize(1);
        _dynamic_array_synapses_2_delay.resize(1);
        _dynamic_array_synapses_2_delay[0] = 0.0011;
        
                        
                        for(int i=0; i<_num__array_synapses_2_sources; i++)
                        {
                            _array_synapses_2_sources[i] = _static_array__array_synapses_2_sources[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_synapses_2_targets; i++)
                        {
                            _array_synapses_2_targets[i] = _static_array__array_synapses_2_targets[i];
                        }
                        
        _run_synapses_2_synapses_create_array_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_2_w_PC_I.size(); i++)
                        {
                            _dynamic_array_synapses_2_w_PC_I[i] = _static_array__dynamic_array_synapses_2_w_PC_I[i];
                        }
                        
        _dynamic_array_synapses_3_delay.resize(1);
        _dynamic_array_synapses_3_delay.resize(1);
        _dynamic_array_synapses_3_delay[0] = 0.0009000000000000001;
        
                        
                        for(int i=0; i<_num__array_synapses_3_sources; i++)
                        {
                            _array_synapses_3_sources[i] = _static_array__array_synapses_3_sources[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_synapses_3_targets; i++)
                        {
                            _array_synapses_3_targets[i] = _static_array__array_synapses_3_targets[i];
                        }
                        
        _run_synapses_3_synapses_create_array_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_3_w_BC_E.size(); i++)
                        {
                            _dynamic_array_synapses_3_w_BC_E[i] = _static_array__dynamic_array_synapses_3_w_BC_E[i];
                        }
                        
        _dynamic_array_synapses_4_delay.resize(1);
        _dynamic_array_synapses_4_delay.resize(1);
        _dynamic_array_synapses_4_delay[0] = 0.0006;
        
                        
                        for(int i=0; i<_num__array_synapses_4_sources; i++)
                        {
                            _array_synapses_4_sources[i] = _static_array__array_synapses_4_sources[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_synapses_4_targets; i++)
                        {
                            _array_synapses_4_targets[i] = _static_array__array_synapses_4_targets[i];
                        }
                        
        _run_synapses_4_synapses_create_array_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_4_w_BC_I.size(); i++)
                        {
                            _dynamic_array_synapses_4_w_BC_I[i] = _static_array__dynamic_array_synapses_4_w_BC_I[i];
                        }
                        
        _run_synapses_5_synapses_create_generator_codeobject();
        _dynamic_array_synapses_6_delay.resize(1);
        _dynamic_array_synapses_6_delay.resize(1);
        _dynamic_array_synapses_6_delay[0] = 0.002227670402342876;
        
                        
                        for(int i=0; i<_num__array_synapses_6_sources; i++)
                        {
                            _array_synapses_6_sources[i] = _static_array__array_synapses_6_sources[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_synapses_6_targets; i++)
                        {
                            _array_synapses_6_targets[i] = _static_array__array_synapses_6_targets[i];
                        }
                        
        _run_synapses_6_synapses_create_array_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_6_w_exc.size(); i++)
                        {
                            _dynamic_array_synapses_6_w_exc[i] = _static_array__dynamic_array_synapses_6_w_exc[i];
                        }
                        
        _array_statemonitor_clock_timestep[0] = 0;
        _array_statemonitor_clock_dt[0] = 0.001;
        _array_statemonitor_clock_dt[0] = 0.001;
        
                        
                        for(int i=0; i<_num__array_statemonitor__indices; i++)
                        {
                            _array_statemonitor__indices[i] = _static_array__array_statemonitor__indices[i];
                        }
                        
        _array_statemonitor_clock_timestep[0] = 0;
        _array_statemonitor_clock_t[0] = 0.0;
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        _before_run_synapses_1_pre_push_spikes();
        _before_run_synapses_2_pre_push_spikes();
        _before_run_synapses_3_pre_push_spikes();
        _before_run_synapses_4_pre_push_spikes();
        _before_run_synapses_5_pre_push_spikes();
        _before_run_synapses_6_pre_push_spikes();
        _before_run_synapses_pre_push_spikes();
        _before_run_synapses_1_post_push_spikes();
        _before_run_synapses_2_post_push_spikes();
        _before_run_synapses_3_post_push_spikes();
        _before_run_synapses_4_post_push_spikes();
        _before_run_synapses_6_post_push_spikes();
        network.clear();
        network.add(&statemonitor_clock, _run_statemonitor_codeobject);
        network.add(&defaultclock, _run_neurongroup_1_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
        network.add(&defaultclock, _run_neurongroup_1_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_neurongroup_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_poissongroup_1_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_poissongroup_spike_thresholder_codeobject);
        network.add(&defaultclock, _run_spikemonitor_codeobject);
        network.add(&defaultclock, _run_spikemonitor_1_codeobject);
        network.add(&defaultclock, _run_synapses_1_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_1_pre_codeobject);
        network.add(&defaultclock, _run_synapses_2_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_2_pre_codeobject);
        network.add(&defaultclock, _run_synapses_3_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_3_pre_codeobject);
        network.add(&defaultclock, _run_synapses_4_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_4_pre_codeobject);
        network.add(&defaultclock, _run_synapses_5_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_5_pre_codeobject);
        network.add(&defaultclock, _run_synapses_6_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_6_pre_codeobject);
        network.add(&defaultclock, _run_synapses_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_pre_codeobject);
        network.add(&defaultclock, _run_synapses_1_post_push_spikes);
        network.add(&defaultclock, _run_synapses_1_post_codeobject);
        network.add(&defaultclock, _run_synapses_2_post_push_spikes);
        network.add(&defaultclock, _run_synapses_2_post_codeobject);
        network.add(&defaultclock, _run_synapses_3_post_push_spikes);
        network.add(&defaultclock, _run_synapses_3_post_codeobject);
        network.add(&defaultclock, _run_synapses_4_post_push_spikes);
        network.add(&defaultclock, _run_synapses_4_post_codeobject);
        network.add(&defaultclock, _run_synapses_6_post_push_spikes);
        network.add(&defaultclock, _run_synapses_6_post_codeobject);
        network.add(&defaultclock, _run_neurongroup_1_spike_resetter_codeobject);
        network.add(&defaultclock, _run_neurongroup_spike_resetter_codeobject);
        network.add(&defaultclock, _run_ratemonitor_codeobject);
        network.add(&defaultclock, _run_ratemonitor_1_codeobject);
        set_from_command_line(args);
        network.run(1.0, report_progress, 10.0);
        _after_run_neurongroup_1_spike_thresholder_codeobject();
        _after_run_neurongroup_spike_thresholder_codeobject();
        _after_run_poissongroup_1_spike_thresholder_codeobject();
        _after_run_poissongroup_spike_thresholder_codeobject();
        _array_statemonitor_clock_timestep[0] = 1000;
        _array_statemonitor_clock_t[0] = 1.0;
        _array_defaultclock_timestep[0] = 10000;
        _array_defaultclock_t[0] = 1.0;
        _before_run_synapses_1_pre_push_spikes();
        _before_run_synapses_2_pre_push_spikes();
        _before_run_synapses_3_pre_push_spikes();
        _before_run_synapses_4_pre_push_spikes();
        _before_run_synapses_5_pre_push_spikes();
        _before_run_synapses_6_pre_push_spikes();
        _before_run_synapses_pre_push_spikes();
        _before_run_synapses_1_post_push_spikes();
        _before_run_synapses_2_post_push_spikes();
        _before_run_synapses_3_post_push_spikes();
        _before_run_synapses_4_post_push_spikes();
        _before_run_synapses_6_post_push_spikes();
        network.clear();
        network.add(&statemonitor_clock, _run_statemonitor_codeobject_1);
        network.add(&defaultclock, _run_neurongroup_1_stateupdater_codeobject_1);
        network.add(&defaultclock, _run_neurongroup_stateupdater_codeobject_1);
        network.add(&defaultclock, _run_neurongroup_1_spike_thresholder_codeobject_1);
        network.add(&defaultclock, _run_neurongroup_spike_thresholder_codeobject_1);
        network.add(&defaultclock, _run_poissongroup_1_spike_thresholder_codeobject_1);
        network.add(&defaultclock, _run_poissongroup_spike_thresholder_codeobject_1);
        network.add(&defaultclock, _run_spikemonitor_codeobject_1);
        network.add(&defaultclock, _run_spikemonitor_1_codeobject_1);
        network.add(&defaultclock, _run_synapses_1_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_1_pre_codeobject_1);
        network.add(&defaultclock, _run_synapses_2_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_2_pre_codeobject_1);
        network.add(&defaultclock, _run_synapses_3_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_3_pre_codeobject_1);
        network.add(&defaultclock, _run_synapses_4_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_4_pre_codeobject_1);
        network.add(&defaultclock, _run_synapses_5_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_5_pre_codeobject_1);
        network.add(&defaultclock, _run_synapses_6_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_6_pre_codeobject_1);
        network.add(&defaultclock, _run_synapses_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_pre_codeobject_1);
        network.add(&defaultclock, _run_synapses_1_post_push_spikes);
        network.add(&defaultclock, _run_synapses_1_post_codeobject_1);
        network.add(&defaultclock, _run_synapses_2_post_push_spikes);
        network.add(&defaultclock, _run_synapses_2_post_codeobject_1);
        network.add(&defaultclock, _run_synapses_3_post_push_spikes);
        network.add(&defaultclock, _run_synapses_3_post_codeobject_1);
        network.add(&defaultclock, _run_synapses_4_post_push_spikes);
        network.add(&defaultclock, _run_synapses_4_post_codeobject_1);
        network.add(&defaultclock, _run_synapses_6_post_push_spikes);
        network.add(&defaultclock, _run_synapses_6_post_codeobject_1);
        network.add(&defaultclock, _run_neurongroup_1_spike_resetter_codeobject_1);
        network.add(&defaultclock, _run_neurongroup_spike_resetter_codeobject_1);
        network.add(&defaultclock, _run_ratemonitor_codeobject_1);
        network.add(&defaultclock, _run_ratemonitor_1_codeobject_1);
        network.run(4.0, report_progress, 10.0);
        _after_run_neurongroup_1_spike_thresholder_codeobject_1();
        _after_run_neurongroup_spike_thresholder_codeobject_1();
        _after_run_poissongroup_1_spike_thresholder_codeobject_1();
        _after_run_poissongroup_spike_thresholder_codeobject_1();
        _array_statemonitor_clock_timestep[0] = 5000;
        _array_statemonitor_clock_t[0] = 5.0;
        _array_defaultclock_timestep[0] = 50000;
        _array_defaultclock_t[0] = 5.0;
        _before_run_synapses_1_pre_push_spikes();
        _before_run_synapses_2_pre_push_spikes();
        _before_run_synapses_3_pre_push_spikes();
        _before_run_synapses_4_pre_push_spikes();
        _before_run_synapses_5_pre_push_spikes();
        _before_run_synapses_6_pre_push_spikes();
        _before_run_synapses_pre_push_spikes();
        _before_run_synapses_1_post_push_spikes();
        _before_run_synapses_2_post_push_spikes();
        _before_run_synapses_3_post_push_spikes();
        _before_run_synapses_4_post_push_spikes();
        _before_run_synapses_6_post_push_spikes();
        network.clear();
        network.add(&statemonitor_clock, _run_statemonitor_codeobject_2);
        network.add(&defaultclock, _run_neurongroup_1_stateupdater_codeobject_2);
        network.add(&defaultclock, _run_neurongroup_stateupdater_codeobject_2);
        network.add(&defaultclock, _run_neurongroup_1_spike_thresholder_codeobject_2);
        network.add(&defaultclock, _run_neurongroup_spike_thresholder_codeobject_2);
        network.add(&defaultclock, _run_poissongroup_1_spike_thresholder_codeobject_2);
        network.add(&defaultclock, _run_poissongroup_spike_thresholder_codeobject_2);
        network.add(&defaultclock, _run_spikemonitor_codeobject_2);
        network.add(&defaultclock, _run_spikemonitor_1_codeobject_2);
        network.add(&defaultclock, _run_synapses_1_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_1_pre_codeobject_2);
        network.add(&defaultclock, _run_synapses_2_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_2_pre_codeobject_2);
        network.add(&defaultclock, _run_synapses_3_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_3_pre_codeobject_2);
        network.add(&defaultclock, _run_synapses_4_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_4_pre_codeobject_2);
        network.add(&defaultclock, _run_synapses_5_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_5_pre_codeobject_2);
        network.add(&defaultclock, _run_synapses_6_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_6_pre_codeobject_2);
        network.add(&defaultclock, _run_synapses_pre_push_spikes);
        network.add(&defaultclock, _run_synapses_pre_codeobject_2);
        network.add(&defaultclock, _run_synapses_1_post_push_spikes);
        network.add(&defaultclock, _run_synapses_1_post_codeobject_2);
        network.add(&defaultclock, _run_synapses_2_post_push_spikes);
        network.add(&defaultclock, _run_synapses_2_post_codeobject_2);
        network.add(&defaultclock, _run_synapses_3_post_push_spikes);
        network.add(&defaultclock, _run_synapses_3_post_codeobject_2);
        network.add(&defaultclock, _run_synapses_4_post_push_spikes);
        network.add(&defaultclock, _run_synapses_4_post_codeobject_2);
        network.add(&defaultclock, _run_synapses_6_post_push_spikes);
        network.add(&defaultclock, _run_synapses_6_post_codeobject_2);
        network.add(&defaultclock, _run_neurongroup_1_spike_resetter_codeobject_2);
        network.add(&defaultclock, _run_neurongroup_spike_resetter_codeobject_2);
        network.add(&defaultclock, _run_ratemonitor_codeobject_2);
        network.add(&defaultclock, _run_ratemonitor_1_codeobject_2);
        network.run(10.0, report_progress, 10.0);
        _after_run_neurongroup_1_spike_thresholder_codeobject_2();
        _after_run_neurongroup_spike_thresholder_codeobject_2();
        _after_run_poissongroup_1_spike_thresholder_codeobject_2();
        _after_run_poissongroup_spike_thresholder_codeobject_2();
        #ifdef DEBUG
        _debugmsg_spikemonitor_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_spikemonitor_1_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_1_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_2_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_3_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_4_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_5_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_6_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_1_post_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_2_post_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_3_post_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_4_post_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_6_post_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_spikemonitor_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_spikemonitor_1_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_1_pre_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_2_pre_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_3_pre_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_4_pre_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_5_pre_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_6_pre_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_pre_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_1_post_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_2_post_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_3_post_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_4_post_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_6_post_codeobject_1();
        #endif
        
        #ifdef DEBUG
        _debugmsg_spikemonitor_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_spikemonitor_1_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_1_pre_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_2_pre_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_3_pre_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_4_pre_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_5_pre_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_6_pre_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_pre_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_1_post_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_2_post_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_3_post_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_4_post_codeobject_2();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_6_post_codeobject_2();
        #endif

	}
        

	brian_end();
        

	return 0;
}