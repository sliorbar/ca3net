
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<chrono>
#include<random>
#include<vector>


namespace brian {

extern std::string results_dir;

class RandomGenerator {
    private:
        std::mt19937 gen;
        double stored_gauss;
        bool has_stored_gauss = false;
    public:
        RandomGenerator() {
            seed();
        }
        void seed() {
            std::random_device rd;
            gen.seed(rd());
            has_stored_gauss = false;
        }
        void seed(unsigned long seed) {
            gen.seed(seed);
            has_stored_gauss = false;
        }
        // Allow exporting/setting the internal state of the random generator
        friend std::ostream& operator<<(std::ostream& out, const RandomGenerator& rng);
        friend std::istream& operator>>(std::istream& in, RandomGenerator& rng);

        double rand() {
            /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
            const long a = gen() >> 5;
            const long b = gen() >> 6;
            return (a * 67108864.0 + b) / 9007199254740992.0;
        }

        double randn() {
            if (has_stored_gauss) {
                const double tmp = stored_gauss;
                has_stored_gauss = false;
                return tmp;
            }
            else {
                double f, x1, x2, r2;

                do {
                    x1 = 2.0*rand() - 1.0;
                    x2 = 2.0*rand() - 1.0;
                    r2 = x1*x1 + x2*x2;
                }
                while (r2 >= 1.0 || r2 == 0.0);

                /* Box-Muller transform */
                f = sqrt(-2.0*log(r2)/r2);
                /* Keep for next call */
                stored_gauss = f*x1;
                has_stored_gauss = true;
                return f*x2;
            }
        }
};

extern std::ostream& operator<<(std::ostream& out, const RandomGenerator& rng);
extern std::istream& operator>>(std::istream& in, RandomGenerator& rng);

// In OpenMP we need one state per thread
extern std::vector< RandomGenerator > _random_generators;

//////////////// clocks ///////////////////
extern Clock defaultclock;
extern Clock statemonitor_clock;

//////////////// networks /////////////////
extern Network network;



void set_variable_by_name(std::string, std::string);

//////////////// dynamic arrays ///////////
extern std::vector<double> _dynamic_array_ratemonitor_1_rate;
extern std::vector<double> _dynamic_array_ratemonitor_1_t;
extern std::vector<double> _dynamic_array_ratemonitor_rate;
extern std::vector<double> _dynamic_array_ratemonitor_t;
extern std::vector<int32_t> _dynamic_array_spikemonitor_1_i;
extern std::vector<double> _dynamic_array_spikemonitor_1_t;
extern std::vector<int32_t> _dynamic_array_spikemonitor_i;
extern std::vector<double> _dynamic_array_spikemonitor_t;
extern std::vector<double> _dynamic_array_statemonitor_t;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_1_Apostsyn;
extern std::vector<double> _dynamic_array_synapses_1_Apresyn;
extern std::vector<double> _dynamic_array_synapses_1_delay;
extern std::vector<double> _dynamic_array_synapses_1_delay_1;
extern std::vector<double> _dynamic_array_synapses_1_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_1_w_exc;
extern std::vector<int32_t> _dynamic_array_synapses_2__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_2__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_2_Apostsyn_PC_I;
extern std::vector<double> _dynamic_array_synapses_2_Apresyn_PC_I;
extern std::vector<double> _dynamic_array_synapses_2_delay;
extern std::vector<double> _dynamic_array_synapses_2_delay_1;
extern std::vector<double> _dynamic_array_synapses_2_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_2_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_2_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_2_w_PC_I;
extern std::vector<int32_t> _dynamic_array_synapses_3__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_3__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_3_Apostsyn_BC_E;
extern std::vector<double> _dynamic_array_synapses_3_Apresyn_BC_E;
extern std::vector<double> _dynamic_array_synapses_3_delay;
extern std::vector<double> _dynamic_array_synapses_3_delay_1;
extern std::vector<double> _dynamic_array_synapses_3_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_3_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_3_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_3_w_BC_E;
extern std::vector<int32_t> _dynamic_array_synapses_4__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_4__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_4_Apostsyn_BC_I;
extern std::vector<double> _dynamic_array_synapses_4_Apresyn_BC_I;
extern std::vector<double> _dynamic_array_synapses_4_delay;
extern std::vector<double> _dynamic_array_synapses_4_delay_1;
extern std::vector<double> _dynamic_array_synapses_4_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_4_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_4_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_4_w_BC_I;
extern std::vector<int32_t> _dynamic_array_synapses_5__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_5__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_5_delay;
extern std::vector<int32_t> _dynamic_array_synapses_5_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_5_N_outgoing;
extern std::vector<int32_t> _dynamic_array_synapses_6__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_6__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_6_Apostsyn;
extern std::vector<double> _dynamic_array_synapses_6_Apresyn;
extern std::vector<double> _dynamic_array_synapses_6_delay;
extern std::vector<double> _dynamic_array_synapses_6_delay_1;
extern std::vector<double> _dynamic_array_synapses_6_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_6_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_6_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_6_w_exc;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_delay;
extern std::vector<int32_t> _dynamic_array_synapses_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_N_outgoing;

//////////////// arrays ///////////////////
extern double *_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double *_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t *_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t *_array_neurongroup_1__spikespace;
extern const int _num__array_neurongroup_1__spikespace;
extern double *_array_neurongroup_1_g_ampa;
extern const int _num__array_neurongroup_1_g_ampa;
extern double *_array_neurongroup_1_g_gaba;
extern const int _num__array_neurongroup_1_g_gaba;
extern int32_t *_array_neurongroup_1_i;
extern const int _num__array_neurongroup_1_i;
extern double *_array_neurongroup_1_lastspike;
extern const int _num__array_neurongroup_1_lastspike;
extern char *_array_neurongroup_1_not_refractory;
extern const int _num__array_neurongroup_1_not_refractory;
extern double *_array_neurongroup_1_vm;
extern const int _num__array_neurongroup_1_vm;
extern double *_array_neurongroup_1_w;
extern const int _num__array_neurongroup_1_w;
extern double *_array_neurongroup_1_x_ampa;
extern const int _num__array_neurongroup_1_x_ampa;
extern double *_array_neurongroup_1_x_gaba;
extern const int _num__array_neurongroup_1_x_gaba;
extern int32_t *_array_neurongroup__spikespace;
extern const int _num__array_neurongroup__spikespace;
extern double *_array_neurongroup_g_ampa;
extern const int _num__array_neurongroup_g_ampa;
extern double *_array_neurongroup_g_ampaMF;
extern const int _num__array_neurongroup_g_ampaMF;
extern double *_array_neurongroup_g_gaba;
extern const int _num__array_neurongroup_g_gaba;
extern int32_t *_array_neurongroup_i;
extern const int _num__array_neurongroup_i;
extern double *_array_neurongroup_lastspike;
extern const int _num__array_neurongroup_lastspike;
extern char *_array_neurongroup_not_refractory;
extern const int _num__array_neurongroup_not_refractory;
extern double *_array_neurongroup_vm;
extern const int _num__array_neurongroup_vm;
extern double *_array_neurongroup_w;
extern const int _num__array_neurongroup_w;
extern double *_array_neurongroup_x_ampa;
extern const int _num__array_neurongroup_x_ampa;
extern double *_array_neurongroup_x_ampaMF;
extern const int _num__array_neurongroup_x_ampaMF;
extern double *_array_neurongroup_x_gaba;
extern const int _num__array_neurongroup_x_gaba;
extern int32_t *_array_poissongroup_1__spikespace;
extern const int _num__array_poissongroup_1__spikespace;
extern int32_t *_array_poissongroup_1_i;
extern const int _num__array_poissongroup_1_i;
extern double *_array_poissongroup_1_rates;
extern const int _num__array_poissongroup_1_rates;
extern int32_t *_array_poissongroup__spikespace;
extern const int _num__array_poissongroup__spikespace;
extern int32_t *_array_poissongroup_i;
extern const int _num__array_poissongroup_i;
extern double *_array_poissongroup_rates;
extern const int _num__array_poissongroup_rates;
extern int32_t *_array_ratemonitor_1_N;
extern const int _num__array_ratemonitor_1_N;
extern int32_t *_array_ratemonitor_N;
extern const int _num__array_ratemonitor_N;
extern int32_t *_array_spikemonitor_1__source_idx;
extern const int _num__array_spikemonitor_1__source_idx;
extern int32_t *_array_spikemonitor_1_count;
extern const int _num__array_spikemonitor_1_count;
extern int32_t *_array_spikemonitor_1_N;
extern const int _num__array_spikemonitor_1_N;
extern int32_t *_array_spikemonitor__source_idx;
extern const int _num__array_spikemonitor__source_idx;
extern int32_t *_array_spikemonitor_count;
extern const int _num__array_spikemonitor_count;
extern int32_t *_array_spikemonitor_N;
extern const int _num__array_spikemonitor_N;
extern int32_t *_array_statemonitor__indices;
extern const int _num__array_statemonitor__indices;
extern double *_array_statemonitor_clock_dt;
extern const int _num__array_statemonitor_clock_dt;
extern double *_array_statemonitor_clock_t;
extern const int _num__array_statemonitor_clock_t;
extern int64_t *_array_statemonitor_clock_timestep;
extern const int _num__array_statemonitor_clock_timestep;
extern int32_t *_array_statemonitor_N;
extern const int _num__array_statemonitor_N;
extern double *_array_statemonitor_w_exc;
extern const int _num__array_statemonitor_w_exc;
extern int32_t *_array_synapses_1_N;
extern const int _num__array_synapses_1_N;
extern int32_t *_array_synapses_1_sources;
extern const int _num__array_synapses_1_sources;
extern int32_t *_array_synapses_1_targets;
extern const int _num__array_synapses_1_targets;
extern int32_t *_array_synapses_2_N;
extern const int _num__array_synapses_2_N;
extern int32_t *_array_synapses_2_sources;
extern const int _num__array_synapses_2_sources;
extern int32_t *_array_synapses_2_targets;
extern const int _num__array_synapses_2_targets;
extern int32_t *_array_synapses_3_N;
extern const int _num__array_synapses_3_N;
extern int32_t *_array_synapses_3_sources;
extern const int _num__array_synapses_3_sources;
extern int32_t *_array_synapses_3_targets;
extern const int _num__array_synapses_3_targets;
extern int32_t *_array_synapses_4_N;
extern const int _num__array_synapses_4_N;
extern int32_t *_array_synapses_4_sources;
extern const int _num__array_synapses_4_sources;
extern int32_t *_array_synapses_4_targets;
extern const int _num__array_synapses_4_targets;
extern int32_t *_array_synapses_5_N;
extern const int _num__array_synapses_5_N;
extern int32_t *_array_synapses_6_N;
extern const int _num__array_synapses_6_N;
extern int32_t *_array_synapses_6_sources;
extern const int _num__array_synapses_6_sources;
extern int32_t *_array_synapses_6_targets;
extern const int _num__array_synapses_6_targets;
extern int32_t *_array_synapses_N;
extern const int _num__array_synapses_N;

//////////////// dynamic arrays 2d /////////
extern DynamicArray2D<double> _dynamic_array_statemonitor_w_exc;

/////////////// static arrays /////////////
extern int32_t *_static_array__array_statemonitor__indices;
extern const int _num__static_array__array_statemonitor__indices;
extern int32_t *_static_array__array_synapses_1_sources;
extern const int _num__static_array__array_synapses_1_sources;
extern int32_t *_static_array__array_synapses_1_targets;
extern const int _num__static_array__array_synapses_1_targets;
extern int32_t *_static_array__array_synapses_2_sources;
extern const int _num__static_array__array_synapses_2_sources;
extern int32_t *_static_array__array_synapses_2_targets;
extern const int _num__static_array__array_synapses_2_targets;
extern int32_t *_static_array__array_synapses_3_sources;
extern const int _num__static_array__array_synapses_3_sources;
extern int32_t *_static_array__array_synapses_3_targets;
extern const int _num__static_array__array_synapses_3_targets;
extern int32_t *_static_array__array_synapses_4_sources;
extern const int _num__static_array__array_synapses_4_sources;
extern int32_t *_static_array__array_synapses_4_targets;
extern const int _num__static_array__array_synapses_4_targets;
extern int32_t *_static_array__array_synapses_6_sources;
extern const int _num__static_array__array_synapses_6_sources;
extern int32_t *_static_array__array_synapses_6_targets;
extern const int _num__static_array__array_synapses_6_targets;
extern double *_static_array__dynamic_array_synapses_1_w_exc;
extern const int _num__static_array__dynamic_array_synapses_1_w_exc;
extern double *_static_array__dynamic_array_synapses_2_w_PC_I;
extern const int _num__static_array__dynamic_array_synapses_2_w_PC_I;
extern double *_static_array__dynamic_array_synapses_3_w_BC_E;
extern const int _num__static_array__dynamic_array_synapses_3_w_BC_E;
extern double *_static_array__dynamic_array_synapses_4_w_BC_I;
extern const int _num__static_array__dynamic_array_synapses_4_w_BC_I;
extern double *_static_array__dynamic_array_synapses_6_w_exc;
extern const int _num__static_array__dynamic_array_synapses_6_w_exc;

//////////////// synapses /////////////////
// synapses
extern SynapticPathway synapses_pre;
// synapses_1
extern SynapticPathway synapses_1_post;
extern SynapticPathway synapses_1_pre;
// synapses_2
extern SynapticPathway synapses_2_post;
extern SynapticPathway synapses_2_pre;
// synapses_3
extern SynapticPathway synapses_3_post;
extern SynapticPathway synapses_3_pre;
// synapses_4
extern SynapticPathway synapses_4_post;
extern SynapticPathway synapses_4_pre;
// synapses_5
extern SynapticPathway synapses_5_pre;
// synapses_6
extern SynapticPathway synapses_6_post;
extern SynapticPathway synapses_6_pre;

// Profiling information for each code object
}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


