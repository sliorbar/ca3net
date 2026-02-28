

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<chrono>
#include<random>
#include<vector>
#include<iostream>
#include<fstream>
#include<map>
#include<tuple>
#include<cstdlib>
#include<string>

namespace brian {

std::string results_dir = "results/";  // can be overwritten by --results_dir command line arg

// For multhreading, we need one generator for each thread.
std::vector< RandomGenerator > _random_generators;

std::ostream& operator<<(std::ostream& out, const RandomGenerator& rng)
{
    return out << rng.gen;
}

std::istream& operator>>(std::istream& in, RandomGenerator& rng)
{
    return in >> rng.gen;
}

//////////////// networks /////////////////
Network network;

void set_variable_from_value(std::string varname, char* var_pointer, size_t size, char value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << (value == 1 ? "True" : "False") << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_value(std::string varname, T* var_pointer, size_t size, T value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << value << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_file(std::string varname, T* var_pointer, size_t data_size, std::string filename) {
    ifstream f;
    streampos size;
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' from file '" << filename << "'" << std::endl;
    #endif
    f.open(filename, ios::in | ios::binary | ios::ate);
    size = f.tellg();
    if (size != data_size) {
        std::cerr << "Error reading '" << filename << "': file size " << size << " does not match expected size " << data_size << std::endl;
        return;
    }
    f.seekg(0, ios::beg);
    if (f.is_open())
        f.read(reinterpret_cast<char *>(var_pointer), data_size);
    else
        std::cerr << "Could not read '" << filename << "'" << std::endl;
    if (f.fail())
        std::cerr << "Error reading '" << filename << "'" << std::endl;
}

//////////////// set arrays by name ///////
void set_variable_by_name(std::string name, std::string s_value) {
    size_t var_size;
    size_t data_size;
    // C-style or Python-style capitalization is allowed for boolean values
    if (s_value == "true" || s_value == "True")
        s_value = "1";
    else if (s_value == "false" || s_value == "False")
        s_value = "0";
    // non-dynamic arrays
    if (name == "neurongroup_1._spikespace") {
        var_size = 301;
        data_size = 301*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_neurongroup_1__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.g_ampa") {
        var_size = 300;
        data_size = 300*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_g_ampa, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_g_ampa, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.g_gaba") {
        var_size = 300;
        data_size = 300*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_g_gaba, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_g_gaba, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.lastspike") {
        var_size = 300;
        data_size = 300*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_lastspike, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_lastspike, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.not_refractory") {
        var_size = 300;
        data_size = 300*sizeof(char);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value(name, _array_neurongroup_1_not_refractory, var_size, (char)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_not_refractory, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.vm") {
        var_size = 300;
        data_size = 300*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_vm, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_vm, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.w") {
        var_size = 300;
        data_size = 300*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_w, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_w, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.x_ampa") {
        var_size = 300;
        data_size = 300*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_x_ampa, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_x_ampa, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.x_gaba") {
        var_size = 300;
        data_size = 300*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_x_gaba, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_x_gaba, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup._spikespace") {
        var_size = 8001;
        data_size = 8001*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_neurongroup__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.g_ampa") {
        var_size = 8000;
        data_size = 8000*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_g_ampa, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_g_ampa, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.g_ampaMF") {
        var_size = 8000;
        data_size = 8000*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_g_ampaMF, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_g_ampaMF, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.g_gaba") {
        var_size = 8000;
        data_size = 8000*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_g_gaba, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_g_gaba, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.lastspike") {
        var_size = 8000;
        data_size = 8000*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_lastspike, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_lastspike, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.not_refractory") {
        var_size = 8000;
        data_size = 8000*sizeof(char);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value(name, _array_neurongroup_not_refractory, var_size, (char)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_not_refractory, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.vm") {
        var_size = 8000;
        data_size = 8000*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_vm, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_vm, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.w") {
        var_size = 8000;
        data_size = 8000*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_w, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_w, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.x_ampa") {
        var_size = 8000;
        data_size = 8000*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_x_ampa, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_x_ampa, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.x_ampaMF") {
        var_size = 8000;
        data_size = 8000*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_x_ampaMF, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_x_ampaMF, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.x_gaba") {
        var_size = 8000;
        data_size = 8000*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_x_gaba, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_x_gaba, data_size, s_value);
        }
        return;
    }
    if (name == "poissongroup_1._spikespace") {
        var_size = 61;
        data_size = 61*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_poissongroup_1__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_poissongroup_1__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "poissongroup_1.rates") {
        var_size = 60;
        data_size = 60*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_poissongroup_1_rates, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_poissongroup_1_rates, data_size, s_value);
        }
        return;
    }
    if (name == "poissongroup._spikespace") {
        var_size = 8001;
        data_size = 8001*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_poissongroup__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_poissongroup__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "poissongroup.rates") {
        var_size = 8000;
        data_size = 8000*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_poissongroup_rates, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_poissongroup_rates, data_size, s_value);
        }
        return;
    }
    // dynamic arrays (1d)
    if (name == "synapses_1.Apostsyn") {
        var_size = _dynamic_array_synapses_1_Apostsyn.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_1_Apostsyn[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_1_Apostsyn[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_1.Apresyn") {
        var_size = _dynamic_array_synapses_1_Apresyn.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_1_Apresyn[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_1_Apresyn[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_1.delay") {
        var_size = _dynamic_array_synapses_1_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_1_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_1_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_1.delay") {
        var_size = _dynamic_array_synapses_1_delay_1.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_1_delay_1[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_1_delay_1[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_1.lastupdate") {
        var_size = _dynamic_array_synapses_1_lastupdate.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_1_lastupdate[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_1_lastupdate[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_1.w_exc") {
        var_size = _dynamic_array_synapses_1_w_exc.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_1_w_exc[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_1_w_exc[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.Apostsyn_PC_I") {
        var_size = _dynamic_array_synapses_2_Apostsyn_PC_I.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_Apostsyn_PC_I[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_Apostsyn_PC_I[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.Apresyn_PC_I") {
        var_size = _dynamic_array_synapses_2_Apresyn_PC_I.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_Apresyn_PC_I[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_Apresyn_PC_I[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.delay") {
        var_size = _dynamic_array_synapses_2_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.delay") {
        var_size = _dynamic_array_synapses_2_delay_1.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_delay_1[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_delay_1[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.lastupdate") {
        var_size = _dynamic_array_synapses_2_lastupdate.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_lastupdate[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_lastupdate[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_2.w_PC_I") {
        var_size = _dynamic_array_synapses_2_w_PC_I.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_w_PC_I[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_w_PC_I[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_3.Apostsyn_BC_E") {
        var_size = _dynamic_array_synapses_3_Apostsyn_BC_E.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_3_Apostsyn_BC_E[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_3_Apostsyn_BC_E[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_3.Apresyn_BC_E") {
        var_size = _dynamic_array_synapses_3_Apresyn_BC_E.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_3_Apresyn_BC_E[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_3_Apresyn_BC_E[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_3.delay") {
        var_size = _dynamic_array_synapses_3_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_3_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_3_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_3.delay") {
        var_size = _dynamic_array_synapses_3_delay_1.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_3_delay_1[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_3_delay_1[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_3.lastupdate") {
        var_size = _dynamic_array_synapses_3_lastupdate.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_3_lastupdate[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_3_lastupdate[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_3.w_BC_E") {
        var_size = _dynamic_array_synapses_3_w_BC_E.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_3_w_BC_E[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_3_w_BC_E[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_4.Apostsyn_BC_I") {
        var_size = _dynamic_array_synapses_4_Apostsyn_BC_I.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_4_Apostsyn_BC_I[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_4_Apostsyn_BC_I[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_4.Apresyn_BC_I") {
        var_size = _dynamic_array_synapses_4_Apresyn_BC_I.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_4_Apresyn_BC_I[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_4_Apresyn_BC_I[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_4.delay") {
        var_size = _dynamic_array_synapses_4_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_4_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_4_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_4.delay") {
        var_size = _dynamic_array_synapses_4_delay_1.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_4_delay_1[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_4_delay_1[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_4.lastupdate") {
        var_size = _dynamic_array_synapses_4_lastupdate.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_4_lastupdate[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_4_lastupdate[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_4.w_BC_I") {
        var_size = _dynamic_array_synapses_4_w_BC_I.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_4_w_BC_I[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_4_w_BC_I[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_5.delay") {
        var_size = _dynamic_array_synapses_5_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_5_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_5_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_6.Apostsyn") {
        var_size = _dynamic_array_synapses_6_Apostsyn.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_6_Apostsyn[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_6_Apostsyn[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_6.Apresyn") {
        var_size = _dynamic_array_synapses_6_Apresyn.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_6_Apresyn[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_6_Apresyn[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_6.delay") {
        var_size = _dynamic_array_synapses_6_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_6_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_6_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_6.delay") {
        var_size = _dynamic_array_synapses_6_delay_1.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_6_delay_1[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_6_delay_1[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_6.lastupdate") {
        var_size = _dynamic_array_synapses_6_lastupdate.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_6_lastupdate[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_6_lastupdate[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_6.w_exc") {
        var_size = _dynamic_array_synapses_6_w_exc.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_6_w_exc[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_6_w_exc[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses.delay") {
        var_size = _dynamic_array_synapses_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_delay[0], data_size, s_value);
        }
        return;
    }
    std::cerr << "Cannot set unknown variable '" << name << "'." << std::endl;
    exit(1);
}
//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_neurongroup_1__spikespace;
const int _num__array_neurongroup_1__spikespace = 301;
double * _array_neurongroup_1_g_ampa;
const int _num__array_neurongroup_1_g_ampa = 300;
double * _array_neurongroup_1_g_gaba;
const int _num__array_neurongroup_1_g_gaba = 300;
int32_t * _array_neurongroup_1_i;
const int _num__array_neurongroup_1_i = 300;
double * _array_neurongroup_1_lastspike;
const int _num__array_neurongroup_1_lastspike = 300;
char * _array_neurongroup_1_not_refractory;
const int _num__array_neurongroup_1_not_refractory = 300;
double * _array_neurongroup_1_vm;
const int _num__array_neurongroup_1_vm = 300;
double * _array_neurongroup_1_w;
const int _num__array_neurongroup_1_w = 300;
double * _array_neurongroup_1_x_ampa;
const int _num__array_neurongroup_1_x_ampa = 300;
double * _array_neurongroup_1_x_gaba;
const int _num__array_neurongroup_1_x_gaba = 300;
int32_t * _array_neurongroup__spikespace;
const int _num__array_neurongroup__spikespace = 8001;
double * _array_neurongroup_g_ampa;
const int _num__array_neurongroup_g_ampa = 8000;
double * _array_neurongroup_g_ampaMF;
const int _num__array_neurongroup_g_ampaMF = 8000;
double * _array_neurongroup_g_gaba;
const int _num__array_neurongroup_g_gaba = 8000;
int32_t * _array_neurongroup_i;
const int _num__array_neurongroup_i = 8000;
double * _array_neurongroup_lastspike;
const int _num__array_neurongroup_lastspike = 8000;
char * _array_neurongroup_not_refractory;
const int _num__array_neurongroup_not_refractory = 8000;
double * _array_neurongroup_vm;
const int _num__array_neurongroup_vm = 8000;
double * _array_neurongroup_w;
const int _num__array_neurongroup_w = 8000;
double * _array_neurongroup_x_ampa;
const int _num__array_neurongroup_x_ampa = 8000;
double * _array_neurongroup_x_ampaMF;
const int _num__array_neurongroup_x_ampaMF = 8000;
double * _array_neurongroup_x_gaba;
const int _num__array_neurongroup_x_gaba = 8000;
int32_t * _array_poissongroup_1__spikespace;
const int _num__array_poissongroup_1__spikespace = 61;
int32_t * _array_poissongroup_1_i;
const int _num__array_poissongroup_1_i = 60;
double * _array_poissongroup_1_rates;
const int _num__array_poissongroup_1_rates = 60;
int32_t * _array_poissongroup__spikespace;
const int _num__array_poissongroup__spikespace = 8001;
int32_t * _array_poissongroup_i;
const int _num__array_poissongroup_i = 8000;
double * _array_poissongroup_rates;
const int _num__array_poissongroup_rates = 8000;
int32_t * _array_ratemonitor_1_N;
const int _num__array_ratemonitor_1_N = 1;
int32_t * _array_ratemonitor_N;
const int _num__array_ratemonitor_N = 1;
int32_t * _array_spikemonitor_1__source_idx;
const int _num__array_spikemonitor_1__source_idx = 300;
int32_t * _array_spikemonitor_1_count;
const int _num__array_spikemonitor_1_count = 300;
int32_t * _array_spikemonitor_1_N;
const int _num__array_spikemonitor_1_N = 1;
int32_t * _array_spikemonitor__source_idx;
const int _num__array_spikemonitor__source_idx = 8000;
int32_t * _array_spikemonitor_count;
const int _num__array_spikemonitor_count = 8000;
int32_t * _array_spikemonitor_N;
const int _num__array_spikemonitor_N = 1;
int32_t * _array_statemonitor__indices;
const int _num__array_statemonitor__indices = 20;
double * _array_statemonitor_clock_dt;
const int _num__array_statemonitor_clock_dt = 1;
double * _array_statemonitor_clock_t;
const int _num__array_statemonitor_clock_t = 1;
int64_t * _array_statemonitor_clock_timestep;
const int _num__array_statemonitor_clock_timestep = 1;
int32_t * _array_statemonitor_N;
const int _num__array_statemonitor_N = 1;
double * _array_statemonitor_w_exc;
const int _num__array_statemonitor_w_exc = (0, 20);
int32_t * _array_synapses_1_N;
const int _num__array_synapses_1_N = 1;
int32_t * _array_synapses_1_sources;
const int _num__array_synapses_1_sources = 6393354;
int32_t * _array_synapses_1_targets;
const int _num__array_synapses_1_targets = 6393354;
int32_t * _array_synapses_2_N;
const int _num__array_synapses_2_N = 1;
int32_t * _array_synapses_2_sources;
const int _num__array_synapses_2_sources = 600374;
int32_t * _array_synapses_2_targets;
const int _num__array_synapses_2_targets = 600374;
int32_t * _array_synapses_3_N;
const int _num__array_synapses_3_N = 1;
int32_t * _array_synapses_3_sources;
const int _num__array_synapses_3_sources = 239691;
int32_t * _array_synapses_3_targets;
const int _num__array_synapses_3_targets = 239691;
int32_t * _array_synapses_4_N;
const int _num__array_synapses_4_N = 1;
int32_t * _array_synapses_4_sources;
const int _num__array_synapses_4_sources = 22444;
int32_t * _array_synapses_4_targets;
const int _num__array_synapses_4_targets = 22444;
int32_t * _array_synapses_5_N;
const int _num__array_synapses_5_N = 1;
int32_t * _array_synapses_6_N;
const int _num__array_synapses_6_N = 1;
int32_t * _array_synapses_6_sources;
const int _num__array_synapses_6_sources = 389;
int32_t * _array_synapses_6_targets;
const int _num__array_synapses_6_targets = 389;
int32_t * _array_synapses_N;
const int _num__array_synapses_N = 1;

//////////////// dynamic arrays 1d /////////
std::vector<double> _dynamic_array_ratemonitor_1_rate;
std::vector<double> _dynamic_array_ratemonitor_1_t;
std::vector<double> _dynamic_array_ratemonitor_rate;
std::vector<double> _dynamic_array_ratemonitor_t;
std::vector<int32_t> _dynamic_array_spikemonitor_1_i;
std::vector<double> _dynamic_array_spikemonitor_1_t;
std::vector<int32_t> _dynamic_array_spikemonitor_i;
std::vector<double> _dynamic_array_spikemonitor_t;
std::vector<double> _dynamic_array_statemonitor_t;
std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
std::vector<double> _dynamic_array_synapses_1_Apostsyn;
std::vector<double> _dynamic_array_synapses_1_Apresyn;
std::vector<double> _dynamic_array_synapses_1_delay;
std::vector<double> _dynamic_array_synapses_1_delay_1;
std::vector<double> _dynamic_array_synapses_1_lastupdate;
std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
std::vector<double> _dynamic_array_synapses_1_w_exc;
std::vector<int32_t> _dynamic_array_synapses_2__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_2__synaptic_pre;
std::vector<double> _dynamic_array_synapses_2_Apostsyn_PC_I;
std::vector<double> _dynamic_array_synapses_2_Apresyn_PC_I;
std::vector<double> _dynamic_array_synapses_2_delay;
std::vector<double> _dynamic_array_synapses_2_delay_1;
std::vector<double> _dynamic_array_synapses_2_lastupdate;
std::vector<int32_t> _dynamic_array_synapses_2_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_2_N_outgoing;
std::vector<double> _dynamic_array_synapses_2_w_PC_I;
std::vector<int32_t> _dynamic_array_synapses_3__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_3__synaptic_pre;
std::vector<double> _dynamic_array_synapses_3_Apostsyn_BC_E;
std::vector<double> _dynamic_array_synapses_3_Apresyn_BC_E;
std::vector<double> _dynamic_array_synapses_3_delay;
std::vector<double> _dynamic_array_synapses_3_delay_1;
std::vector<double> _dynamic_array_synapses_3_lastupdate;
std::vector<int32_t> _dynamic_array_synapses_3_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_3_N_outgoing;
std::vector<double> _dynamic_array_synapses_3_w_BC_E;
std::vector<int32_t> _dynamic_array_synapses_4__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_4__synaptic_pre;
std::vector<double> _dynamic_array_synapses_4_Apostsyn_BC_I;
std::vector<double> _dynamic_array_synapses_4_Apresyn_BC_I;
std::vector<double> _dynamic_array_synapses_4_delay;
std::vector<double> _dynamic_array_synapses_4_delay_1;
std::vector<double> _dynamic_array_synapses_4_lastupdate;
std::vector<int32_t> _dynamic_array_synapses_4_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_4_N_outgoing;
std::vector<double> _dynamic_array_synapses_4_w_BC_I;
std::vector<int32_t> _dynamic_array_synapses_5__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_5__synaptic_pre;
std::vector<double> _dynamic_array_synapses_5_delay;
std::vector<int32_t> _dynamic_array_synapses_5_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_5_N_outgoing;
std::vector<int32_t> _dynamic_array_synapses_6__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_6__synaptic_pre;
std::vector<double> _dynamic_array_synapses_6_Apostsyn;
std::vector<double> _dynamic_array_synapses_6_Apresyn;
std::vector<double> _dynamic_array_synapses_6_delay;
std::vector<double> _dynamic_array_synapses_6_delay_1;
std::vector<double> _dynamic_array_synapses_6_lastupdate;
std::vector<int32_t> _dynamic_array_synapses_6_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_6_N_outgoing;
std::vector<double> _dynamic_array_synapses_6_w_exc;
std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
std::vector<double> _dynamic_array_synapses_delay;
std::vector<int32_t> _dynamic_array_synapses_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_N_outgoing;

//////////////// dynamic arrays 2d /////////
DynamicArray2D<double> _dynamic_array_statemonitor_w_exc;

/////////////// static arrays /////////////
int32_t * _static_array__array_statemonitor__indices;
const int _num__static_array__array_statemonitor__indices = 20;
int32_t * _static_array__array_synapses_1_sources;
const int _num__static_array__array_synapses_1_sources = 6393354;
int32_t * _static_array__array_synapses_1_targets;
const int _num__static_array__array_synapses_1_targets = 6393354;
int32_t * _static_array__array_synapses_2_sources;
const int _num__static_array__array_synapses_2_sources = 600374;
int32_t * _static_array__array_synapses_2_targets;
const int _num__static_array__array_synapses_2_targets = 600374;
int32_t * _static_array__array_synapses_3_sources;
const int _num__static_array__array_synapses_3_sources = 239691;
int32_t * _static_array__array_synapses_3_targets;
const int _num__static_array__array_synapses_3_targets = 239691;
int32_t * _static_array__array_synapses_4_sources;
const int _num__static_array__array_synapses_4_sources = 22444;
int32_t * _static_array__array_synapses_4_targets;
const int _num__static_array__array_synapses_4_targets = 22444;
int32_t * _static_array__array_synapses_6_sources;
const int _num__static_array__array_synapses_6_sources = 389;
int32_t * _static_array__array_synapses_6_targets;
const int _num__static_array__array_synapses_6_targets = 389;
double * _static_array__dynamic_array_synapses_1_w_exc;
const int _num__static_array__dynamic_array_synapses_1_w_exc = 6393354;
double * _static_array__dynamic_array_synapses_2_w_PC_I;
const int _num__static_array__dynamic_array_synapses_2_w_PC_I = 600374;
double * _static_array__dynamic_array_synapses_3_w_BC_E;
const int _num__static_array__dynamic_array_synapses_3_w_BC_E = 239691;
double * _static_array__dynamic_array_synapses_4_w_BC_I;
const int _num__static_array__dynamic_array_synapses_4_w_BC_I = 22444;
double * _static_array__dynamic_array_synapses_6_w_exc;
const int _num__static_array__dynamic_array_synapses_6_w_exc = 389;

//////////////// synapses /////////////////
// synapses
SynapticPathway synapses_pre(
    _dynamic_array_synapses__synaptic_pre,
    0, 8000);
// synapses_1
SynapticPathway synapses_1_post(
    _dynamic_array_synapses_1__synaptic_post,
    0, 8000);
SynapticPathway synapses_1_pre(
    _dynamic_array_synapses_1__synaptic_pre,
    0, 8000);
// synapses_2
SynapticPathway synapses_2_post(
    _dynamic_array_synapses_2__synaptic_post,
    0, 300);
SynapticPathway synapses_2_pre(
    _dynamic_array_synapses_2__synaptic_pre,
    0, 8000);
// synapses_3
SynapticPathway synapses_3_post(
    _dynamic_array_synapses_3__synaptic_post,
    0, 8000);
SynapticPathway synapses_3_pre(
    _dynamic_array_synapses_3__synaptic_pre,
    0, 300);
// synapses_4
SynapticPathway synapses_4_post(
    _dynamic_array_synapses_4__synaptic_post,
    0, 300);
SynapticPathway synapses_4_pre(
    _dynamic_array_synapses_4__synaptic_pre,
    0, 300);
// synapses_5
SynapticPathway synapses_5_pre(
    _dynamic_array_synapses_5__synaptic_pre,
    0, 60);
// synapses_6
SynapticPathway synapses_6_post(
    _dynamic_array_synapses_6__synaptic_post,
    0, 8000);
SynapticPathway synapses_6_pre(
    _dynamic_array_synapses_6__synaptic_pre,
    0, 60);

//////////////// clocks ///////////////////
// attributes will be set in run.cpp
Clock defaultclock;
Clock statemonitor_clock;

// Profiling information for each code object
}

void _init_arrays()
{
    using namespace brian;

    // Arrays initialized to 0
    _array_defaultclock_dt = new double[1];
    
    for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;

    _array_defaultclock_t = new double[1];
    
    for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;

    _array_defaultclock_timestep = new int64_t[1];
    
    for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;

    _array_neurongroup_1__spikespace = new int32_t[301];
    
    for(int i=0; i<301; i++) _array_neurongroup_1__spikespace[i] = 0;

    _array_neurongroup_1_g_ampa = new double[300];
    
    for(int i=0; i<300; i++) _array_neurongroup_1_g_ampa[i] = 0;

    _array_neurongroup_1_g_gaba = new double[300];
    
    for(int i=0; i<300; i++) _array_neurongroup_1_g_gaba[i] = 0;

    _array_neurongroup_1_i = new int32_t[300];
    
    for(int i=0; i<300; i++) _array_neurongroup_1_i[i] = 0;

    _array_neurongroup_1_lastspike = new double[300];
    
    for(int i=0; i<300; i++) _array_neurongroup_1_lastspike[i] = 0;

    _array_neurongroup_1_not_refractory = new char[300];
    
    for(int i=0; i<300; i++) _array_neurongroup_1_not_refractory[i] = 0;

    _array_neurongroup_1_vm = new double[300];
    
    for(int i=0; i<300; i++) _array_neurongroup_1_vm[i] = 0;

    _array_neurongroup_1_w = new double[300];
    
    for(int i=0; i<300; i++) _array_neurongroup_1_w[i] = 0;

    _array_neurongroup_1_x_ampa = new double[300];
    
    for(int i=0; i<300; i++) _array_neurongroup_1_x_ampa[i] = 0;

    _array_neurongroup_1_x_gaba = new double[300];
    
    for(int i=0; i<300; i++) _array_neurongroup_1_x_gaba[i] = 0;

    _array_neurongroup__spikespace = new int32_t[8001];
    
    for(int i=0; i<8001; i++) _array_neurongroup__spikespace[i] = 0;

    _array_neurongroup_g_ampa = new double[8000];
    
    for(int i=0; i<8000; i++) _array_neurongroup_g_ampa[i] = 0;

    _array_neurongroup_g_ampaMF = new double[8000];
    
    for(int i=0; i<8000; i++) _array_neurongroup_g_ampaMF[i] = 0;

    _array_neurongroup_g_gaba = new double[8000];
    
    for(int i=0; i<8000; i++) _array_neurongroup_g_gaba[i] = 0;

    _array_neurongroup_i = new int32_t[8000];
    
    for(int i=0; i<8000; i++) _array_neurongroup_i[i] = 0;

    _array_neurongroup_lastspike = new double[8000];
    
    for(int i=0; i<8000; i++) _array_neurongroup_lastspike[i] = 0;

    _array_neurongroup_not_refractory = new char[8000];
    
    for(int i=0; i<8000; i++) _array_neurongroup_not_refractory[i] = 0;

    _array_neurongroup_vm = new double[8000];
    
    for(int i=0; i<8000; i++) _array_neurongroup_vm[i] = 0;

    _array_neurongroup_w = new double[8000];
    
    for(int i=0; i<8000; i++) _array_neurongroup_w[i] = 0;

    _array_neurongroup_x_ampa = new double[8000];
    
    for(int i=0; i<8000; i++) _array_neurongroup_x_ampa[i] = 0;

    _array_neurongroup_x_ampaMF = new double[8000];
    
    for(int i=0; i<8000; i++) _array_neurongroup_x_ampaMF[i] = 0;

    _array_neurongroup_x_gaba = new double[8000];
    
    for(int i=0; i<8000; i++) _array_neurongroup_x_gaba[i] = 0;

    _array_poissongroup_1__spikespace = new int32_t[61];
    
    for(int i=0; i<61; i++) _array_poissongroup_1__spikespace[i] = 0;

    _array_poissongroup_1_i = new int32_t[60];
    
    for(int i=0; i<60; i++) _array_poissongroup_1_i[i] = 0;

    _array_poissongroup_1_rates = new double[60];
    
    for(int i=0; i<60; i++) _array_poissongroup_1_rates[i] = 0;

    _array_poissongroup__spikespace = new int32_t[8001];
    
    for(int i=0; i<8001; i++) _array_poissongroup__spikespace[i] = 0;

    _array_poissongroup_i = new int32_t[8000];
    
    for(int i=0; i<8000; i++) _array_poissongroup_i[i] = 0;

    _array_poissongroup_rates = new double[8000];
    
    for(int i=0; i<8000; i++) _array_poissongroup_rates[i] = 0;

    _array_ratemonitor_1_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_ratemonitor_1_N[i] = 0;

    _array_ratemonitor_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_ratemonitor_N[i] = 0;

    _array_spikemonitor_1__source_idx = new int32_t[300];
    
    for(int i=0; i<300; i++) _array_spikemonitor_1__source_idx[i] = 0;

    _array_spikemonitor_1_count = new int32_t[300];
    
    for(int i=0; i<300; i++) _array_spikemonitor_1_count[i] = 0;

    _array_spikemonitor_1_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_spikemonitor_1_N[i] = 0;

    _array_spikemonitor__source_idx = new int32_t[8000];
    
    for(int i=0; i<8000; i++) _array_spikemonitor__source_idx[i] = 0;

    _array_spikemonitor_count = new int32_t[8000];
    
    for(int i=0; i<8000; i++) _array_spikemonitor_count[i] = 0;

    _array_spikemonitor_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;

    _array_statemonitor__indices = new int32_t[20];
    
    for(int i=0; i<20; i++) _array_statemonitor__indices[i] = 0;

    _array_statemonitor_clock_dt = new double[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_clock_dt[i] = 0;

    _array_statemonitor_clock_t = new double[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_clock_t[i] = 0;

    _array_statemonitor_clock_timestep = new int64_t[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_clock_timestep[i] = 0;

    _array_statemonitor_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_N[i] = 0;

    _array_synapses_1_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;

    _array_synapses_1_sources = new int32_t[6393354];
    
    for(int i=0; i<6393354; i++) _array_synapses_1_sources[i] = 0;

    _array_synapses_1_targets = new int32_t[6393354];
    
    for(int i=0; i<6393354; i++) _array_synapses_1_targets[i] = 0;

    _array_synapses_2_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_synapses_2_N[i] = 0;

    _array_synapses_2_sources = new int32_t[600374];
    
    for(int i=0; i<600374; i++) _array_synapses_2_sources[i] = 0;

    _array_synapses_2_targets = new int32_t[600374];
    
    for(int i=0; i<600374; i++) _array_synapses_2_targets[i] = 0;

    _array_synapses_3_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_synapses_3_N[i] = 0;

    _array_synapses_3_sources = new int32_t[239691];
    
    for(int i=0; i<239691; i++) _array_synapses_3_sources[i] = 0;

    _array_synapses_3_targets = new int32_t[239691];
    
    for(int i=0; i<239691; i++) _array_synapses_3_targets[i] = 0;

    _array_synapses_4_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_synapses_4_N[i] = 0;

    _array_synapses_4_sources = new int32_t[22444];
    
    for(int i=0; i<22444; i++) _array_synapses_4_sources[i] = 0;

    _array_synapses_4_targets = new int32_t[22444];
    
    for(int i=0; i<22444; i++) _array_synapses_4_targets[i] = 0;

    _array_synapses_5_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_synapses_5_N[i] = 0;

    _array_synapses_6_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_synapses_6_N[i] = 0;

    _array_synapses_6_sources = new int32_t[389];
    
    for(int i=0; i<389; i++) _array_synapses_6_sources[i] = 0;

    _array_synapses_6_targets = new int32_t[389];
    
    for(int i=0; i<389; i++) _array_synapses_6_targets[i] = 0;

    _array_synapses_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_synapses_N[i] = 0;

    _dynamic_array_synapses_1_delay.resize(1);
    
    for(int i=0; i<1; i++) _dynamic_array_synapses_1_delay[i] = 0;

    _dynamic_array_synapses_2_delay.resize(1);
    
    for(int i=0; i<1; i++) _dynamic_array_synapses_2_delay[i] = 0;

    _dynamic_array_synapses_3_delay.resize(1);
    
    for(int i=0; i<1; i++) _dynamic_array_synapses_3_delay[i] = 0;

    _dynamic_array_synapses_4_delay.resize(1);
    
    for(int i=0; i<1; i++) _dynamic_array_synapses_4_delay[i] = 0;

    _dynamic_array_synapses_6_delay.resize(1);
    
    for(int i=0; i<1; i++) _dynamic_array_synapses_6_delay[i] = 0;


    // Arrays initialized to an "arange"
    _array_neurongroup_1_i = new int32_t[300];
    
    for(int i=0; i<300; i++) _array_neurongroup_1_i[i] = 0 + i;

    _array_neurongroup_i = new int32_t[8000];
    
    for(int i=0; i<8000; i++) _array_neurongroup_i[i] = 0 + i;

    _array_poissongroup_1_i = new int32_t[60];
    
    for(int i=0; i<60; i++) _array_poissongroup_1_i[i] = 0 + i;

    _array_poissongroup_i = new int32_t[8000];
    
    for(int i=0; i<8000; i++) _array_poissongroup_i[i] = 0 + i;

    _array_spikemonitor_1__source_idx = new int32_t[300];
    
    for(int i=0; i<300; i++) _array_spikemonitor_1__source_idx[i] = 0 + i;

    _array_spikemonitor__source_idx = new int32_t[8000];
    
    for(int i=0; i<8000; i++) _array_spikemonitor__source_idx[i] = 0 + i;


    // static arrays
    _static_array__array_statemonitor__indices = new int32_t[20];
    _static_array__array_synapses_1_sources = new int32_t[6393354];
    _static_array__array_synapses_1_targets = new int32_t[6393354];
    _static_array__array_synapses_2_sources = new int32_t[600374];
    _static_array__array_synapses_2_targets = new int32_t[600374];
    _static_array__array_synapses_3_sources = new int32_t[239691];
    _static_array__array_synapses_3_targets = new int32_t[239691];
    _static_array__array_synapses_4_sources = new int32_t[22444];
    _static_array__array_synapses_4_targets = new int32_t[22444];
    _static_array__array_synapses_6_sources = new int32_t[389];
    _static_array__array_synapses_6_targets = new int32_t[389];
    _static_array__dynamic_array_synapses_1_w_exc = new double[6393354];
    _static_array__dynamic_array_synapses_2_w_PC_I = new double[600374];
    _static_array__dynamic_array_synapses_3_w_BC_E = new double[239691];
    _static_array__dynamic_array_synapses_4_w_BC_I = new double[22444];
    _static_array__dynamic_array_synapses_6_w_exc = new double[389];

    // Random number generator states
    std::random_device rd;
    for (int i=0; i<1; i++)
        _random_generators.push_back(RandomGenerator());
}

void _load_arrays()
{
    using namespace brian;

    ifstream f_static_array__array_statemonitor__indices;
    f_static_array__array_statemonitor__indices.open("static_arrays/_static_array__array_statemonitor__indices", ios::in | ios::binary);
    if(f_static_array__array_statemonitor__indices.is_open())
    {
        f_static_array__array_statemonitor__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor__indices), 20*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_statemonitor__indices." << endl;
    }
    ifstream f_static_array__array_synapses_1_sources;
    f_static_array__array_synapses_1_sources.open("static_arrays/_static_array__array_synapses_1_sources", ios::in | ios::binary);
    if(f_static_array__array_synapses_1_sources.is_open())
    {
        f_static_array__array_synapses_1_sources.read(reinterpret_cast<char*>(_static_array__array_synapses_1_sources), 6393354*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_1_sources." << endl;
    }
    ifstream f_static_array__array_synapses_1_targets;
    f_static_array__array_synapses_1_targets.open("static_arrays/_static_array__array_synapses_1_targets", ios::in | ios::binary);
    if(f_static_array__array_synapses_1_targets.is_open())
    {
        f_static_array__array_synapses_1_targets.read(reinterpret_cast<char*>(_static_array__array_synapses_1_targets), 6393354*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_1_targets." << endl;
    }
    ifstream f_static_array__array_synapses_2_sources;
    f_static_array__array_synapses_2_sources.open("static_arrays/_static_array__array_synapses_2_sources", ios::in | ios::binary);
    if(f_static_array__array_synapses_2_sources.is_open())
    {
        f_static_array__array_synapses_2_sources.read(reinterpret_cast<char*>(_static_array__array_synapses_2_sources), 600374*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_2_sources." << endl;
    }
    ifstream f_static_array__array_synapses_2_targets;
    f_static_array__array_synapses_2_targets.open("static_arrays/_static_array__array_synapses_2_targets", ios::in | ios::binary);
    if(f_static_array__array_synapses_2_targets.is_open())
    {
        f_static_array__array_synapses_2_targets.read(reinterpret_cast<char*>(_static_array__array_synapses_2_targets), 600374*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_2_targets." << endl;
    }
    ifstream f_static_array__array_synapses_3_sources;
    f_static_array__array_synapses_3_sources.open("static_arrays/_static_array__array_synapses_3_sources", ios::in | ios::binary);
    if(f_static_array__array_synapses_3_sources.is_open())
    {
        f_static_array__array_synapses_3_sources.read(reinterpret_cast<char*>(_static_array__array_synapses_3_sources), 239691*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_3_sources." << endl;
    }
    ifstream f_static_array__array_synapses_3_targets;
    f_static_array__array_synapses_3_targets.open("static_arrays/_static_array__array_synapses_3_targets", ios::in | ios::binary);
    if(f_static_array__array_synapses_3_targets.is_open())
    {
        f_static_array__array_synapses_3_targets.read(reinterpret_cast<char*>(_static_array__array_synapses_3_targets), 239691*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_3_targets." << endl;
    }
    ifstream f_static_array__array_synapses_4_sources;
    f_static_array__array_synapses_4_sources.open("static_arrays/_static_array__array_synapses_4_sources", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_sources.is_open())
    {
        f_static_array__array_synapses_4_sources.read(reinterpret_cast<char*>(_static_array__array_synapses_4_sources), 22444*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_sources." << endl;
    }
    ifstream f_static_array__array_synapses_4_targets;
    f_static_array__array_synapses_4_targets.open("static_arrays/_static_array__array_synapses_4_targets", ios::in | ios::binary);
    if(f_static_array__array_synapses_4_targets.is_open())
    {
        f_static_array__array_synapses_4_targets.read(reinterpret_cast<char*>(_static_array__array_synapses_4_targets), 22444*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_4_targets." << endl;
    }
    ifstream f_static_array__array_synapses_6_sources;
    f_static_array__array_synapses_6_sources.open("static_arrays/_static_array__array_synapses_6_sources", ios::in | ios::binary);
    if(f_static_array__array_synapses_6_sources.is_open())
    {
        f_static_array__array_synapses_6_sources.read(reinterpret_cast<char*>(_static_array__array_synapses_6_sources), 389*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_6_sources." << endl;
    }
    ifstream f_static_array__array_synapses_6_targets;
    f_static_array__array_synapses_6_targets.open("static_arrays/_static_array__array_synapses_6_targets", ios::in | ios::binary);
    if(f_static_array__array_synapses_6_targets.is_open())
    {
        f_static_array__array_synapses_6_targets.read(reinterpret_cast<char*>(_static_array__array_synapses_6_targets), 389*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_6_targets." << endl;
    }
    ifstream f_static_array__dynamic_array_synapses_1_w_exc;
    f_static_array__dynamic_array_synapses_1_w_exc.open("static_arrays/_static_array__dynamic_array_synapses_1_w_exc", ios::in | ios::binary);
    if(f_static_array__dynamic_array_synapses_1_w_exc.is_open())
    {
        f_static_array__dynamic_array_synapses_1_w_exc.read(reinterpret_cast<char*>(_static_array__dynamic_array_synapses_1_w_exc), 6393354*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_synapses_1_w_exc." << endl;
    }
    ifstream f_static_array__dynamic_array_synapses_2_w_PC_I;
    f_static_array__dynamic_array_synapses_2_w_PC_I.open("static_arrays/_static_array__dynamic_array_synapses_2_w_PC_I", ios::in | ios::binary);
    if(f_static_array__dynamic_array_synapses_2_w_PC_I.is_open())
    {
        f_static_array__dynamic_array_synapses_2_w_PC_I.read(reinterpret_cast<char*>(_static_array__dynamic_array_synapses_2_w_PC_I), 600374*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_synapses_2_w_PC_I." << endl;
    }
    ifstream f_static_array__dynamic_array_synapses_3_w_BC_E;
    f_static_array__dynamic_array_synapses_3_w_BC_E.open("static_arrays/_static_array__dynamic_array_synapses_3_w_BC_E", ios::in | ios::binary);
    if(f_static_array__dynamic_array_synapses_3_w_BC_E.is_open())
    {
        f_static_array__dynamic_array_synapses_3_w_BC_E.read(reinterpret_cast<char*>(_static_array__dynamic_array_synapses_3_w_BC_E), 239691*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_synapses_3_w_BC_E." << endl;
    }
    ifstream f_static_array__dynamic_array_synapses_4_w_BC_I;
    f_static_array__dynamic_array_synapses_4_w_BC_I.open("static_arrays/_static_array__dynamic_array_synapses_4_w_BC_I", ios::in | ios::binary);
    if(f_static_array__dynamic_array_synapses_4_w_BC_I.is_open())
    {
        f_static_array__dynamic_array_synapses_4_w_BC_I.read(reinterpret_cast<char*>(_static_array__dynamic_array_synapses_4_w_BC_I), 22444*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_synapses_4_w_BC_I." << endl;
    }
    ifstream f_static_array__dynamic_array_synapses_6_w_exc;
    f_static_array__dynamic_array_synapses_6_w_exc.open("static_arrays/_static_array__dynamic_array_synapses_6_w_exc", ios::in | ios::binary);
    if(f_static_array__dynamic_array_synapses_6_w_exc.is_open())
    {
        f_static_array__dynamic_array_synapses_6_w_exc.read(reinterpret_cast<char*>(_static_array__dynamic_array_synapses_6_w_exc), 389*sizeof(double));
    } else
    {
        std::cout << "Error opening static array _static_array__dynamic_array_synapses_6_w_exc." << endl;
    }
}

void _write_arrays()
{
    using namespace brian;

    ofstream outfile__array_defaultclock_dt;
    outfile__array_defaultclock_dt.open(results_dir + "_array_defaultclock_dt_1978099143", ios::binary | ios::out);
    if(outfile__array_defaultclock_dt.is_open())
    {
        outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
        outfile__array_defaultclock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
    }
    ofstream outfile__array_defaultclock_t;
    outfile__array_defaultclock_t.open(results_dir + "_array_defaultclock_t_2669362164", ios::binary | ios::out);
    if(outfile__array_defaultclock_t.is_open())
    {
        outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
        outfile__array_defaultclock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_t." << endl;
    }
    ofstream outfile__array_defaultclock_timestep;
    outfile__array_defaultclock_timestep.open(results_dir + "_array_defaultclock_timestep_144223508", ios::binary | ios::out);
    if(outfile__array_defaultclock_timestep.is_open())
    {
        outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
        outfile__array_defaultclock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
    }
    ofstream outfile__array_neurongroup_1__spikespace;
    outfile__array_neurongroup_1__spikespace.open(results_dir + "_array_neurongroup_1__spikespace_3155027917", ios::binary | ios::out);
    if(outfile__array_neurongroup_1__spikespace.is_open())
    {
        outfile__array_neurongroup_1__spikespace.write(reinterpret_cast<char*>(_array_neurongroup_1__spikespace), 301*sizeof(_array_neurongroup_1__spikespace[0]));
        outfile__array_neurongroup_1__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1__spikespace." << endl;
    }
    ofstream outfile__array_neurongroup_1_g_ampa;
    outfile__array_neurongroup_1_g_ampa.open(results_dir + "_array_neurongroup_1_g_ampa_1933136302", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_g_ampa.is_open())
    {
        outfile__array_neurongroup_1_g_ampa.write(reinterpret_cast<char*>(_array_neurongroup_1_g_ampa), 300*sizeof(_array_neurongroup_1_g_ampa[0]));
        outfile__array_neurongroup_1_g_ampa.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_g_ampa." << endl;
    }
    ofstream outfile__array_neurongroup_1_g_gaba;
    outfile__array_neurongroup_1_g_gaba.open(results_dir + "_array_neurongroup_1_g_gaba_666666949", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_g_gaba.is_open())
    {
        outfile__array_neurongroup_1_g_gaba.write(reinterpret_cast<char*>(_array_neurongroup_1_g_gaba), 300*sizeof(_array_neurongroup_1_g_gaba[0]));
        outfile__array_neurongroup_1_g_gaba.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_g_gaba." << endl;
    }
    ofstream outfile__array_neurongroup_1_i;
    outfile__array_neurongroup_1_i.open(results_dir + "_array_neurongroup_1_i_3674354357", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_i.is_open())
    {
        outfile__array_neurongroup_1_i.write(reinterpret_cast<char*>(_array_neurongroup_1_i), 300*sizeof(_array_neurongroup_1_i[0]));
        outfile__array_neurongroup_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_i." << endl;
    }
    ofstream outfile__array_neurongroup_1_lastspike;
    outfile__array_neurongroup_1_lastspike.open(results_dir + "_array_neurongroup_1_lastspike_1163579662", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_lastspike.is_open())
    {
        outfile__array_neurongroup_1_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_1_lastspike), 300*sizeof(_array_neurongroup_1_lastspike[0]));
        outfile__array_neurongroup_1_lastspike.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_lastspike." << endl;
    }
    ofstream outfile__array_neurongroup_1_not_refractory;
    outfile__array_neurongroup_1_not_refractory.open(results_dir + "_array_neurongroup_1_not_refractory_897855399", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_not_refractory.is_open())
    {
        outfile__array_neurongroup_1_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_1_not_refractory), 300*sizeof(_array_neurongroup_1_not_refractory[0]));
        outfile__array_neurongroup_1_not_refractory.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_not_refractory." << endl;
    }
    ofstream outfile__array_neurongroup_1_vm;
    outfile__array_neurongroup_1_vm.open(results_dir + "_array_neurongroup_1_vm_2542516679", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_vm.is_open())
    {
        outfile__array_neurongroup_1_vm.write(reinterpret_cast<char*>(_array_neurongroup_1_vm), 300*sizeof(_array_neurongroup_1_vm[0]));
        outfile__array_neurongroup_1_vm.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_vm." << endl;
    }
    ofstream outfile__array_neurongroup_1_w;
    outfile__array_neurongroup_1_w.open(results_dir + "_array_neurongroup_1_w_554504150", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_w.is_open())
    {
        outfile__array_neurongroup_1_w.write(reinterpret_cast<char*>(_array_neurongroup_1_w), 300*sizeof(_array_neurongroup_1_w[0]));
        outfile__array_neurongroup_1_w.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_w." << endl;
    }
    ofstream outfile__array_neurongroup_1_x_ampa;
    outfile__array_neurongroup_1_x_ampa.open(results_dir + "_array_neurongroup_1_x_ampa_2176442848", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_x_ampa.is_open())
    {
        outfile__array_neurongroup_1_x_ampa.write(reinterpret_cast<char*>(_array_neurongroup_1_x_ampa), 300*sizeof(_array_neurongroup_1_x_ampa[0]));
        outfile__array_neurongroup_1_x_ampa.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_x_ampa." << endl;
    }
    ofstream outfile__array_neurongroup_1_x_gaba;
    outfile__array_neurongroup_1_x_gaba.open(results_dir + "_array_neurongroup_1_x_gaba_3577493387", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_x_gaba.is_open())
    {
        outfile__array_neurongroup_1_x_gaba.write(reinterpret_cast<char*>(_array_neurongroup_1_x_gaba), 300*sizeof(_array_neurongroup_1_x_gaba[0]));
        outfile__array_neurongroup_1_x_gaba.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_x_gaba." << endl;
    }
    ofstream outfile__array_neurongroup__spikespace;
    outfile__array_neurongroup__spikespace.open(results_dir + "_array_neurongroup__spikespace_3522821529", ios::binary | ios::out);
    if(outfile__array_neurongroup__spikespace.is_open())
    {
        outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), 8001*sizeof(_array_neurongroup__spikespace[0]));
        outfile__array_neurongroup__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
    }
    ofstream outfile__array_neurongroup_g_ampa;
    outfile__array_neurongroup_g_ampa.open(results_dir + "_array_neurongroup_g_ampa_1967940385", ios::binary | ios::out);
    if(outfile__array_neurongroup_g_ampa.is_open())
    {
        outfile__array_neurongroup_g_ampa.write(reinterpret_cast<char*>(_array_neurongroup_g_ampa), 8000*sizeof(_array_neurongroup_g_ampa[0]));
        outfile__array_neurongroup_g_ampa.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_g_ampa." << endl;
    }
    ofstream outfile__array_neurongroup_g_ampaMF;
    outfile__array_neurongroup_g_ampaMF.open(results_dir + "_array_neurongroup_g_ampaMF_3281360735", ios::binary | ios::out);
    if(outfile__array_neurongroup_g_ampaMF.is_open())
    {
        outfile__array_neurongroup_g_ampaMF.write(reinterpret_cast<char*>(_array_neurongroup_g_ampaMF), 8000*sizeof(_array_neurongroup_g_ampaMF[0]));
        outfile__array_neurongroup_g_ampaMF.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_g_ampaMF." << endl;
    }
    ofstream outfile__array_neurongroup_g_gaba;
    outfile__array_neurongroup_g_gaba.open(results_dir + "_array_neurongroup_g_gaba_566867274", ios::binary | ios::out);
    if(outfile__array_neurongroup_g_gaba.is_open())
    {
        outfile__array_neurongroup_g_gaba.write(reinterpret_cast<char*>(_array_neurongroup_g_gaba), 8000*sizeof(_array_neurongroup_g_gaba[0]));
        outfile__array_neurongroup_g_gaba.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_g_gaba." << endl;
    }
    ofstream outfile__array_neurongroup_i;
    outfile__array_neurongroup_i.open(results_dir + "_array_neurongroup_i_2649026944", ios::binary | ios::out);
    if(outfile__array_neurongroup_i.is_open())
    {
        outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 8000*sizeof(_array_neurongroup_i[0]));
        outfile__array_neurongroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_i." << endl;
    }
    ofstream outfile__array_neurongroup_lastspike;
    outfile__array_neurongroup_lastspike.open(results_dir + "_array_neurongroup_lastspike_1647074423", ios::binary | ios::out);
    if(outfile__array_neurongroup_lastspike.is_open())
    {
        outfile__array_neurongroup_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_lastspike), 8000*sizeof(_array_neurongroup_lastspike[0]));
        outfile__array_neurongroup_lastspike.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_lastspike." << endl;
    }
    ofstream outfile__array_neurongroup_not_refractory;
    outfile__array_neurongroup_not_refractory.open(results_dir + "_array_neurongroup_not_refractory_1422681464", ios::binary | ios::out);
    if(outfile__array_neurongroup_not_refractory.is_open())
    {
        outfile__array_neurongroup_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_not_refractory), 8000*sizeof(_array_neurongroup_not_refractory[0]));
        outfile__array_neurongroup_not_refractory.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_not_refractory." << endl;
    }
    ofstream outfile__array_neurongroup_vm;
    outfile__array_neurongroup_vm.open(results_dir + "_array_neurongroup_vm_3246299943", ios::binary | ios::out);
    if(outfile__array_neurongroup_vm.is_open())
    {
        outfile__array_neurongroup_vm.write(reinterpret_cast<char*>(_array_neurongroup_vm), 8000*sizeof(_array_neurongroup_vm[0]));
        outfile__array_neurongroup_vm.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_vm." << endl;
    }
    ofstream outfile__array_neurongroup_w;
    outfile__array_neurongroup_w.open(results_dir + "_array_neurongroup_w_1743506659", ios::binary | ios::out);
    if(outfile__array_neurongroup_w.is_open())
    {
        outfile__array_neurongroup_w.write(reinterpret_cast<char*>(_array_neurongroup_w), 8000*sizeof(_array_neurongroup_w[0]));
        outfile__array_neurongroup_w.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_w." << endl;
    }
    ofstream outfile__array_neurongroup_x_ampa;
    outfile__array_neurongroup_x_ampa.open(results_dir + "_array_neurongroup_x_ampa_2278347631", ios::binary | ios::out);
    if(outfile__array_neurongroup_x_ampa.is_open())
    {
        outfile__array_neurongroup_x_ampa.write(reinterpret_cast<char*>(_array_neurongroup_x_ampa), 8000*sizeof(_array_neurongroup_x_ampa[0]));
        outfile__array_neurongroup_x_ampa.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_x_ampa." << endl;
    }
    ofstream outfile__array_neurongroup_x_ampaMF;
    outfile__array_neurongroup_x_ampaMF.open(results_dir + "_array_neurongroup_x_ampaMF_1712239832", ios::binary | ios::out);
    if(outfile__array_neurongroup_x_ampaMF.is_open())
    {
        outfile__array_neurongroup_x_ampaMF.write(reinterpret_cast<char*>(_array_neurongroup_x_ampaMF), 8000*sizeof(_array_neurongroup_x_ampaMF[0]));
        outfile__array_neurongroup_x_ampaMF.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_x_ampaMF." << endl;
    }
    ofstream outfile__array_neurongroup_x_gaba;
    outfile__array_neurongroup_x_gaba.open(results_dir + "_array_neurongroup_x_gaba_3544777988", ios::binary | ios::out);
    if(outfile__array_neurongroup_x_gaba.is_open())
    {
        outfile__array_neurongroup_x_gaba.write(reinterpret_cast<char*>(_array_neurongroup_x_gaba), 8000*sizeof(_array_neurongroup_x_gaba[0]));
        outfile__array_neurongroup_x_gaba.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_x_gaba." << endl;
    }
    ofstream outfile__array_poissongroup_1__spikespace;
    outfile__array_poissongroup_1__spikespace.open(results_dir + "_array_poissongroup_1__spikespace_2558380132", ios::binary | ios::out);
    if(outfile__array_poissongroup_1__spikespace.is_open())
    {
        outfile__array_poissongroup_1__spikespace.write(reinterpret_cast<char*>(_array_poissongroup_1__spikespace), 61*sizeof(_array_poissongroup_1__spikespace[0]));
        outfile__array_poissongroup_1__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_poissongroup_1__spikespace." << endl;
    }
    ofstream outfile__array_poissongroup_1_i;
    outfile__array_poissongroup_1_i.open(results_dir + "_array_poissongroup_1_i_2566510749", ios::binary | ios::out);
    if(outfile__array_poissongroup_1_i.is_open())
    {
        outfile__array_poissongroup_1_i.write(reinterpret_cast<char*>(_array_poissongroup_1_i), 60*sizeof(_array_poissongroup_1_i[0]));
        outfile__array_poissongroup_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_poissongroup_1_i." << endl;
    }
    ofstream outfile__array_poissongroup_1_rates;
    outfile__array_poissongroup_1_rates.open(results_dir + "_array_poissongroup_1_rates_3230304882", ios::binary | ios::out);
    if(outfile__array_poissongroup_1_rates.is_open())
    {
        outfile__array_poissongroup_1_rates.write(reinterpret_cast<char*>(_array_poissongroup_1_rates), 60*sizeof(_array_poissongroup_1_rates[0]));
        outfile__array_poissongroup_1_rates.close();
    } else
    {
        std::cout << "Error writing output file for _array_poissongroup_1_rates." << endl;
    }
    ofstream outfile__array_poissongroup__spikespace;
    outfile__array_poissongroup__spikespace.open(results_dir + "_array_poissongroup__spikespace_1019000416", ios::binary | ios::out);
    if(outfile__array_poissongroup__spikespace.is_open())
    {
        outfile__array_poissongroup__spikespace.write(reinterpret_cast<char*>(_array_poissongroup__spikespace), 8001*sizeof(_array_poissongroup__spikespace[0]));
        outfile__array_poissongroup__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_poissongroup__spikespace." << endl;
    }
    ofstream outfile__array_poissongroup_i;
    outfile__array_poissongroup_i.open(results_dir + "_array_poissongroup_i_1277690444", ios::binary | ios::out);
    if(outfile__array_poissongroup_i.is_open())
    {
        outfile__array_poissongroup_i.write(reinterpret_cast<char*>(_array_poissongroup_i), 8000*sizeof(_array_poissongroup_i[0]));
        outfile__array_poissongroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_poissongroup_i." << endl;
    }
    ofstream outfile__array_poissongroup_rates;
    outfile__array_poissongroup_rates.open(results_dir + "_array_poissongroup_rates_3353413371", ios::binary | ios::out);
    if(outfile__array_poissongroup_rates.is_open())
    {
        outfile__array_poissongroup_rates.write(reinterpret_cast<char*>(_array_poissongroup_rates), 8000*sizeof(_array_poissongroup_rates[0]));
        outfile__array_poissongroup_rates.close();
    } else
    {
        std::cout << "Error writing output file for _array_poissongroup_rates." << endl;
    }
    ofstream outfile__array_ratemonitor_1_N;
    outfile__array_ratemonitor_1_N.open(results_dir + "_array_ratemonitor_1_N_1152192744", ios::binary | ios::out);
    if(outfile__array_ratemonitor_1_N.is_open())
    {
        outfile__array_ratemonitor_1_N.write(reinterpret_cast<char*>(_array_ratemonitor_1_N), 1*sizeof(_array_ratemonitor_1_N[0]));
        outfile__array_ratemonitor_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_ratemonitor_1_N." << endl;
    }
    ofstream outfile__array_ratemonitor_N;
    outfile__array_ratemonitor_N.open(results_dir + "_array_ratemonitor_N_611090289", ios::binary | ios::out);
    if(outfile__array_ratemonitor_N.is_open())
    {
        outfile__array_ratemonitor_N.write(reinterpret_cast<char*>(_array_ratemonitor_N), 1*sizeof(_array_ratemonitor_N[0]));
        outfile__array_ratemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_ratemonitor_N." << endl;
    }
    ofstream outfile__array_spikemonitor_1__source_idx;
    outfile__array_spikemonitor_1__source_idx.open(results_dir + "_array_spikemonitor_1__source_idx_3609292218", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1__source_idx.is_open())
    {
        outfile__array_spikemonitor_1__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_1__source_idx), 300*sizeof(_array_spikemonitor_1__source_idx[0]));
        outfile__array_spikemonitor_1__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1__source_idx." << endl;
    }
    ofstream outfile__array_spikemonitor_1_count;
    outfile__array_spikemonitor_1_count.open(results_dir + "_array_spikemonitor_1_count_3862916462", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1_count.is_open())
    {
        outfile__array_spikemonitor_1_count.write(reinterpret_cast<char*>(_array_spikemonitor_1_count), 300*sizeof(_array_spikemonitor_1_count[0]));
        outfile__array_spikemonitor_1_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1_count." << endl;
    }
    ofstream outfile__array_spikemonitor_1_N;
    outfile__array_spikemonitor_1_N.open(results_dir + "_array_spikemonitor_1_N_2390248205", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1_N.is_open())
    {
        outfile__array_spikemonitor_1_N.write(reinterpret_cast<char*>(_array_spikemonitor_1_N), 1*sizeof(_array_spikemonitor_1_N[0]));
        outfile__array_spikemonitor_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1_N." << endl;
    }
    ofstream outfile__array_spikemonitor__source_idx;
    outfile__array_spikemonitor__source_idx.open(results_dir + "_array_spikemonitor__source_idx_1477951789", ios::binary | ios::out);
    if(outfile__array_spikemonitor__source_idx.is_open())
    {
        outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 8000*sizeof(_array_spikemonitor__source_idx[0]));
        outfile__array_spikemonitor__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
    }
    ofstream outfile__array_spikemonitor_count;
    outfile__array_spikemonitor_count.open(results_dir + "_array_spikemonitor_count_598337445", ios::binary | ios::out);
    if(outfile__array_spikemonitor_count.is_open())
    {
        outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 8000*sizeof(_array_spikemonitor_count[0]));
        outfile__array_spikemonitor_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
    }
    ofstream outfile__array_spikemonitor_N;
    outfile__array_spikemonitor_N.open(results_dir + "_array_spikemonitor_N_225734567", ios::binary | ios::out);
    if(outfile__array_spikemonitor_N.is_open())
    {
        outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(_array_spikemonitor_N[0]));
        outfile__array_spikemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
    }
    ofstream outfile__array_statemonitor__indices;
    outfile__array_statemonitor__indices.open(results_dir + "_array_statemonitor__indices_2854283999", ios::binary | ios::out);
    if(outfile__array_statemonitor__indices.is_open())
    {
        outfile__array_statemonitor__indices.write(reinterpret_cast<char*>(_array_statemonitor__indices), 20*sizeof(_array_statemonitor__indices[0]));
        outfile__array_statemonitor__indices.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor__indices." << endl;
    }
    ofstream outfile__array_statemonitor_clock_dt;
    outfile__array_statemonitor_clock_dt.open(results_dir + "_array_statemonitor_clock_dt_1392639074", ios::binary | ios::out);
    if(outfile__array_statemonitor_clock_dt.is_open())
    {
        outfile__array_statemonitor_clock_dt.write(reinterpret_cast<char*>(_array_statemonitor_clock_dt), 1*sizeof(_array_statemonitor_clock_dt[0]));
        outfile__array_statemonitor_clock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_clock_dt." << endl;
    }
    ofstream outfile__array_statemonitor_clock_t;
    outfile__array_statemonitor_clock_t.open(results_dir + "_array_statemonitor_clock_t_2696032964", ios::binary | ios::out);
    if(outfile__array_statemonitor_clock_t.is_open())
    {
        outfile__array_statemonitor_clock_t.write(reinterpret_cast<char*>(_array_statemonitor_clock_t), 1*sizeof(_array_statemonitor_clock_t[0]));
        outfile__array_statemonitor_clock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_clock_t." << endl;
    }
    ofstream outfile__array_statemonitor_clock_timestep;
    outfile__array_statemonitor_clock_timestep.open(results_dir + "_array_statemonitor_clock_timestep_1384860830", ios::binary | ios::out);
    if(outfile__array_statemonitor_clock_timestep.is_open())
    {
        outfile__array_statemonitor_clock_timestep.write(reinterpret_cast<char*>(_array_statemonitor_clock_timestep), 1*sizeof(_array_statemonitor_clock_timestep[0]));
        outfile__array_statemonitor_clock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_clock_timestep." << endl;
    }
    ofstream outfile__array_statemonitor_N;
    outfile__array_statemonitor_N.open(results_dir + "_array_statemonitor_N_4140778434", ios::binary | ios::out);
    if(outfile__array_statemonitor_N.is_open())
    {
        outfile__array_statemonitor_N.write(reinterpret_cast<char*>(_array_statemonitor_N), 1*sizeof(_array_statemonitor_N[0]));
        outfile__array_statemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_N." << endl;
    }
    ofstream outfile__array_synapses_1_N;
    outfile__array_synapses_1_N.open(results_dir + "_array_synapses_1_N_1771729519", ios::binary | ios::out);
    if(outfile__array_synapses_1_N.is_open())
    {
        outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(_array_synapses_1_N[0]));
        outfile__array_synapses_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_1_N." << endl;
    }
    ofstream outfile__array_synapses_1_sources;
    outfile__array_synapses_1_sources.open(results_dir + "_array_synapses_1_sources_93121092", ios::binary | ios::out);
    if(outfile__array_synapses_1_sources.is_open())
    {
        outfile__array_synapses_1_sources.write(reinterpret_cast<char*>(_array_synapses_1_sources), 6393354*sizeof(_array_synapses_1_sources[0]));
        outfile__array_synapses_1_sources.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_1_sources." << endl;
    }
    ofstream outfile__array_synapses_1_targets;
    outfile__array_synapses_1_targets.open(results_dir + "_array_synapses_1_targets_2022871461", ios::binary | ios::out);
    if(outfile__array_synapses_1_targets.is_open())
    {
        outfile__array_synapses_1_targets.write(reinterpret_cast<char*>(_array_synapses_1_targets), 6393354*sizeof(_array_synapses_1_targets[0]));
        outfile__array_synapses_1_targets.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_1_targets." << endl;
    }
    ofstream outfile__array_synapses_2_N;
    outfile__array_synapses_2_N.open(results_dir + "_array_synapses_2_N_1809632310", ios::binary | ios::out);
    if(outfile__array_synapses_2_N.is_open())
    {
        outfile__array_synapses_2_N.write(reinterpret_cast<char*>(_array_synapses_2_N), 1*sizeof(_array_synapses_2_N[0]));
        outfile__array_synapses_2_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_2_N." << endl;
    }
    ofstream outfile__array_synapses_2_sources;
    outfile__array_synapses_2_sources.open(results_dir + "_array_synapses_2_sources_1006753409", ios::binary | ios::out);
    if(outfile__array_synapses_2_sources.is_open())
    {
        outfile__array_synapses_2_sources.write(reinterpret_cast<char*>(_array_synapses_2_sources), 600374*sizeof(_array_synapses_2_sources[0]));
        outfile__array_synapses_2_sources.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_2_sources." << endl;
    }
    ofstream outfile__array_synapses_2_targets;
    outfile__array_synapses_2_targets.open(results_dir + "_array_synapses_2_targets_1092595040", ios::binary | ios::out);
    if(outfile__array_synapses_2_targets.is_open())
    {
        outfile__array_synapses_2_targets.write(reinterpret_cast<char*>(_array_synapses_2_targets), 600374*sizeof(_array_synapses_2_targets[0]));
        outfile__array_synapses_2_targets.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_2_targets." << endl;
    }
    ofstream outfile__array_synapses_3_N;
    outfile__array_synapses_3_N.open(results_dir + "_array_synapses_3_N_1780393473", ios::binary | ios::out);
    if(outfile__array_synapses_3_N.is_open())
    {
        outfile__array_synapses_3_N.write(reinterpret_cast<char*>(_array_synapses_3_N), 1*sizeof(_array_synapses_3_N[0]));
        outfile__array_synapses_3_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_3_N." << endl;
    }
    ofstream outfile__array_synapses_3_sources;
    outfile__array_synapses_3_sources.open(results_dir + "_array_synapses_3_sources_729465538", ios::binary | ios::out);
    if(outfile__array_synapses_3_sources.is_open())
    {
        outfile__array_synapses_3_sources.write(reinterpret_cast<char*>(_array_synapses_3_sources), 239691*sizeof(_array_synapses_3_sources[0]));
        outfile__array_synapses_3_sources.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_3_sources." << endl;
    }
    ofstream outfile__array_synapses_3_targets;
    outfile__array_synapses_3_targets.open(results_dir + "_array_synapses_3_targets_1449441571", ios::binary | ios::out);
    if(outfile__array_synapses_3_targets.is_open())
    {
        outfile__array_synapses_3_targets.write(reinterpret_cast<char*>(_array_synapses_3_targets), 239691*sizeof(_array_synapses_3_targets[0]));
        outfile__array_synapses_3_targets.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_3_targets." << endl;
    }
    ofstream outfile__array_synapses_4_N;
    outfile__array_synapses_4_N.open(results_dir + "_array_synapses_4_N_1867624580", ios::binary | ios::out);
    if(outfile__array_synapses_4_N.is_open())
    {
        outfile__array_synapses_4_N.write(reinterpret_cast<char*>(_array_synapses_4_N), 1*sizeof(_array_synapses_4_N[0]));
        outfile__array_synapses_4_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_N." << endl;
    }
    ofstream outfile__array_synapses_4_sources;
    outfile__array_synapses_4_sources.open(results_dir + "_array_synapses_4_sources_1327214347", ios::binary | ios::out);
    if(outfile__array_synapses_4_sources.is_open())
    {
        outfile__array_synapses_4_sources.write(reinterpret_cast<char*>(_array_synapses_4_sources), 22444*sizeof(_array_synapses_4_sources[0]));
        outfile__array_synapses_4_sources.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_sources." << endl;
    }
    ofstream outfile__array_synapses_4_targets;
    outfile__array_synapses_4_targets.open(results_dir + "_array_synapses_4_targets_839242986", ios::binary | ios::out);
    if(outfile__array_synapses_4_targets.is_open())
    {
        outfile__array_synapses_4_targets.write(reinterpret_cast<char*>(_array_synapses_4_targets), 22444*sizeof(_array_synapses_4_targets[0]));
        outfile__array_synapses_4_targets.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_4_targets." << endl;
    }
    ofstream outfile__array_synapses_5_N;
    outfile__array_synapses_5_N.open(results_dir + "_array_synapses_5_N_1855183539", ios::binary | ios::out);
    if(outfile__array_synapses_5_N.is_open())
    {
        outfile__array_synapses_5_N.write(reinterpret_cast<char*>(_array_synapses_5_N), 1*sizeof(_array_synapses_5_N[0]));
        outfile__array_synapses_5_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_5_N." << endl;
    }
    ofstream outfile__array_synapses_6_N;
    outfile__array_synapses_6_N.open(results_dir + "_array_synapses_6_N_1825924330", ios::binary | ios::out);
    if(outfile__array_synapses_6_N.is_open())
    {
        outfile__array_synapses_6_N.write(reinterpret_cast<char*>(_array_synapses_6_N), 1*sizeof(_array_synapses_6_N[0]));
        outfile__array_synapses_6_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_6_N." << endl;
    }
    ofstream outfile__array_synapses_6_sources;
    outfile__array_synapses_6_sources.open(results_dir + "_array_synapses_6_sources_1642956685", ios::binary | ios::out);
    if(outfile__array_synapses_6_sources.is_open())
    {
        outfile__array_synapses_6_sources.write(reinterpret_cast<char*>(_array_synapses_6_sources), 389*sizeof(_array_synapses_6_sources[0]));
        outfile__array_synapses_6_sources.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_6_sources." << endl;
    }
    ofstream outfile__array_synapses_6_targets;
    outfile__array_synapses_6_targets.open(results_dir + "_array_synapses_6_targets_485751916", ios::binary | ios::out);
    if(outfile__array_synapses_6_targets.is_open())
    {
        outfile__array_synapses_6_targets.write(reinterpret_cast<char*>(_array_synapses_6_targets), 389*sizeof(_array_synapses_6_targets[0]));
        outfile__array_synapses_6_targets.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_6_targets." << endl;
    }
    ofstream outfile__array_synapses_N;
    outfile__array_synapses_N.open(results_dir + "_array_synapses_N_483293785", ios::binary | ios::out);
    if(outfile__array_synapses_N.is_open())
    {
        outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(_array_synapses_N[0]));
        outfile__array_synapses_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_N." << endl;
    }

    ofstream outfile__dynamic_array_ratemonitor_1_rate;
    outfile__dynamic_array_ratemonitor_1_rate.open(results_dir + "_dynamic_array_ratemonitor_1_rate_956123542", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_1_rate.is_open())
    {
        if (! _dynamic_array_ratemonitor_1_rate.empty() )
        {
            outfile__dynamic_array_ratemonitor_1_rate.write(reinterpret_cast<char*>(&_dynamic_array_ratemonitor_1_rate[0]), _dynamic_array_ratemonitor_1_rate.size()*sizeof(_dynamic_array_ratemonitor_1_rate[0]));
            outfile__dynamic_array_ratemonitor_1_rate.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_1_rate." << endl;
    }
    ofstream outfile__dynamic_array_ratemonitor_1_t;
    outfile__dynamic_array_ratemonitor_1_t.open(results_dir + "_dynamic_array_ratemonitor_1_t_991605158", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_1_t.is_open())
    {
        if (! _dynamic_array_ratemonitor_1_t.empty() )
        {
            outfile__dynamic_array_ratemonitor_1_t.write(reinterpret_cast<char*>(&_dynamic_array_ratemonitor_1_t[0]), _dynamic_array_ratemonitor_1_t.size()*sizeof(_dynamic_array_ratemonitor_1_t[0]));
            outfile__dynamic_array_ratemonitor_1_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_1_t." << endl;
    }
    ofstream outfile__dynamic_array_ratemonitor_rate;
    outfile__dynamic_array_ratemonitor_rate.open(results_dir + "_dynamic_array_ratemonitor_rate_1996511615", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_rate.is_open())
    {
        if (! _dynamic_array_ratemonitor_rate.empty() )
        {
            outfile__dynamic_array_ratemonitor_rate.write(reinterpret_cast<char*>(&_dynamic_array_ratemonitor_rate[0]), _dynamic_array_ratemonitor_rate.size()*sizeof(_dynamic_array_ratemonitor_rate[0]));
            outfile__dynamic_array_ratemonitor_rate.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_rate." << endl;
    }
    ofstream outfile__dynamic_array_ratemonitor_t;
    outfile__dynamic_array_ratemonitor_t.open(results_dir + "_dynamic_array_ratemonitor_t_1139349932", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_t.is_open())
    {
        if (! _dynamic_array_ratemonitor_t.empty() )
        {
            outfile__dynamic_array_ratemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_ratemonitor_t[0]), _dynamic_array_ratemonitor_t.size()*sizeof(_dynamic_array_ratemonitor_t[0]));
            outfile__dynamic_array_ratemonitor_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_spikemonitor_1_i;
    outfile__dynamic_array_spikemonitor_1_i.open(results_dir + "_dynamic_array_spikemonitor_1_i_2680224553", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_1_i.is_open())
    {
        if (! _dynamic_array_spikemonitor_1_i.empty() )
        {
            outfile__dynamic_array_spikemonitor_1_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_1_i[0]), _dynamic_array_spikemonitor_1_i.size()*sizeof(_dynamic_array_spikemonitor_1_i[0]));
            outfile__dynamic_array_spikemonitor_1_i.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_i." << endl;
    }
    ofstream outfile__dynamic_array_spikemonitor_1_t;
    outfile__dynamic_array_spikemonitor_1_t.open(results_dir + "_dynamic_array_spikemonitor_1_t_4240873456", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_1_t.is_open())
    {
        if (! _dynamic_array_spikemonitor_1_t.empty() )
        {
            outfile__dynamic_array_spikemonitor_1_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_1_t[0]), _dynamic_array_spikemonitor_1_t.size()*sizeof(_dynamic_array_spikemonitor_1_t[0]));
            outfile__dynamic_array_spikemonitor_1_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_t." << endl;
    }
    ofstream outfile__dynamic_array_spikemonitor_i;
    outfile__dynamic_array_spikemonitor_i.open(results_dir + "_dynamic_array_spikemonitor_i_1976709050", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_i.is_open())
    {
        if (! _dynamic_array_spikemonitor_i.empty() )
        {
            outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_i[0]), _dynamic_array_spikemonitor_i.size()*sizeof(_dynamic_array_spikemonitor_i[0]));
            outfile__dynamic_array_spikemonitor_i.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
    }
    ofstream outfile__dynamic_array_spikemonitor_t;
    outfile__dynamic_array_spikemonitor_t.open(results_dir + "_dynamic_array_spikemonitor_t_383009635", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_t.is_open())
    {
        if (! _dynamic_array_spikemonitor_t.empty() )
        {
            outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_t[0]), _dynamic_array_spikemonitor_t.size()*sizeof(_dynamic_array_spikemonitor_t[0]));
            outfile__dynamic_array_spikemonitor_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_statemonitor_t;
    outfile__dynamic_array_statemonitor_t.open(results_dir + "_dynamic_array_statemonitor_t_3983503110", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_t.is_open())
    {
        if (! _dynamic_array_statemonitor_t.empty() )
        {
            outfile__dynamic_array_statemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_t[0]), _dynamic_array_statemonitor_t.size()*sizeof(_dynamic_array_statemonitor_t[0]));
            outfile__dynamic_array_statemonitor_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_post;
    outfile__dynamic_array_synapses_1__synaptic_post.open(results_dir + "_dynamic_array_synapses_1__synaptic_post_1999337987", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
    {
        if (! _dynamic_array_synapses_1__synaptic_post.empty() )
        {
            outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_post[0]), _dynamic_array_synapses_1__synaptic_post.size()*sizeof(_dynamic_array_synapses_1__synaptic_post[0]));
            outfile__dynamic_array_synapses_1__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
    outfile__dynamic_array_synapses_1__synaptic_pre.open(results_dir + "_dynamic_array_synapses_1__synaptic_pre_681065502", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
    {
        if (! _dynamic_array_synapses_1__synaptic_pre.empty() )
        {
            outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_pre[0]), _dynamic_array_synapses_1__synaptic_pre.size()*sizeof(_dynamic_array_synapses_1__synaptic_pre[0]));
            outfile__dynamic_array_synapses_1__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_Apostsyn;
    outfile__dynamic_array_synapses_1_Apostsyn.open(results_dir + "_dynamic_array_synapses_1_Apostsyn_3434851865", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_Apostsyn.is_open())
    {
        if (! _dynamic_array_synapses_1_Apostsyn.empty() )
        {
            outfile__dynamic_array_synapses_1_Apostsyn.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_Apostsyn[0]), _dynamic_array_synapses_1_Apostsyn.size()*sizeof(_dynamic_array_synapses_1_Apostsyn[0]));
            outfile__dynamic_array_synapses_1_Apostsyn.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_Apostsyn." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_Apresyn;
    outfile__dynamic_array_synapses_1_Apresyn.open(results_dir + "_dynamic_array_synapses_1_Apresyn_700636228", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_Apresyn.is_open())
    {
        if (! _dynamic_array_synapses_1_Apresyn.empty() )
        {
            outfile__dynamic_array_synapses_1_Apresyn.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_Apresyn[0]), _dynamic_array_synapses_1_Apresyn.size()*sizeof(_dynamic_array_synapses_1_Apresyn[0]));
            outfile__dynamic_array_synapses_1_Apresyn.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_Apresyn." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_delay;
    outfile__dynamic_array_synapses_1_delay.open(results_dir + "_dynamic_array_synapses_1_delay_2373823482", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_delay.is_open())
    {
        if (! _dynamic_array_synapses_1_delay.empty() )
        {
            outfile__dynamic_array_synapses_1_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_delay[0]), _dynamic_array_synapses_1_delay.size()*sizeof(_dynamic_array_synapses_1_delay[0]));
            outfile__dynamic_array_synapses_1_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_delay_1;
    outfile__dynamic_array_synapses_1_delay_1.open(results_dir + "_dynamic_array_synapses_1_delay_1_2188619124", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_delay_1.is_open())
    {
        if (! _dynamic_array_synapses_1_delay_1.empty() )
        {
            outfile__dynamic_array_synapses_1_delay_1.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_delay_1[0]), _dynamic_array_synapses_1_delay_1.size()*sizeof(_dynamic_array_synapses_1_delay_1[0]));
            outfile__dynamic_array_synapses_1_delay_1.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_delay_1." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_lastupdate;
    outfile__dynamic_array_synapses_1_lastupdate.open(results_dir + "_dynamic_array_synapses_1_lastupdate_1464104228", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_lastupdate.is_open())
    {
        if (! _dynamic_array_synapses_1_lastupdate.empty() )
        {
            outfile__dynamic_array_synapses_1_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_lastupdate[0]), _dynamic_array_synapses_1_lastupdate.size()*sizeof(_dynamic_array_synapses_1_lastupdate[0]));
            outfile__dynamic_array_synapses_1_lastupdate.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_lastupdate." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_N_incoming;
    outfile__dynamic_array_synapses_1_N_incoming.open(results_dir + "_dynamic_array_synapses_1_N_incoming_3469555706", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
    {
        if (! _dynamic_array_synapses_1_N_incoming.empty() )
        {
            outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_incoming[0]), _dynamic_array_synapses_1_N_incoming.size()*sizeof(_dynamic_array_synapses_1_N_incoming[0]));
            outfile__dynamic_array_synapses_1_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_N_outgoing;
    outfile__dynamic_array_synapses_1_N_outgoing.open(results_dir + "_dynamic_array_synapses_1_N_outgoing_3922806560", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
    {
        if (! _dynamic_array_synapses_1_N_outgoing.empty() )
        {
            outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_outgoing[0]), _dynamic_array_synapses_1_N_outgoing.size()*sizeof(_dynamic_array_synapses_1_N_outgoing[0]));
            outfile__dynamic_array_synapses_1_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_w_exc;
    outfile__dynamic_array_synapses_1_w_exc.open(results_dir + "_dynamic_array_synapses_1_w_exc_1545090432", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_w_exc.is_open())
    {
        if (! _dynamic_array_synapses_1_w_exc.empty() )
        {
            outfile__dynamic_array_synapses_1_w_exc.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_w_exc[0]), _dynamic_array_synapses_1_w_exc.size()*sizeof(_dynamic_array_synapses_1_w_exc[0]));
            outfile__dynamic_array_synapses_1_w_exc.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_w_exc." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_post;
    outfile__dynamic_array_synapses_2__synaptic_post.open(results_dir + "_dynamic_array_synapses_2__synaptic_post_1591987953", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_post.is_open())
    {
        if (! _dynamic_array_synapses_2__synaptic_post.empty() )
        {
            outfile__dynamic_array_synapses_2__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2__synaptic_post[0]), _dynamic_array_synapses_2__synaptic_post.size()*sizeof(_dynamic_array_synapses_2__synaptic_post[0]));
            outfile__dynamic_array_synapses_2__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_pre;
    outfile__dynamic_array_synapses_2__synaptic_pre.open(results_dir + "_dynamic_array_synapses_2__synaptic_pre_971331175", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_pre.is_open())
    {
        if (! _dynamic_array_synapses_2__synaptic_pre.empty() )
        {
            outfile__dynamic_array_synapses_2__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2__synaptic_pre[0]), _dynamic_array_synapses_2__synaptic_pre.size()*sizeof(_dynamic_array_synapses_2__synaptic_pre[0]));
            outfile__dynamic_array_synapses_2__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_Apostsyn_PC_I;
    outfile__dynamic_array_synapses_2_Apostsyn_PC_I.open(results_dir + "_dynamic_array_synapses_2_Apostsyn_PC_I_2297320348", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_Apostsyn_PC_I.is_open())
    {
        if (! _dynamic_array_synapses_2_Apostsyn_PC_I.empty() )
        {
            outfile__dynamic_array_synapses_2_Apostsyn_PC_I.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_Apostsyn_PC_I[0]), _dynamic_array_synapses_2_Apostsyn_PC_I.size()*sizeof(_dynamic_array_synapses_2_Apostsyn_PC_I[0]));
            outfile__dynamic_array_synapses_2_Apostsyn_PC_I.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_Apostsyn_PC_I." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_Apresyn_PC_I;
    outfile__dynamic_array_synapses_2_Apresyn_PC_I.open(results_dir + "_dynamic_array_synapses_2_Apresyn_PC_I_2624465701", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_Apresyn_PC_I.is_open())
    {
        if (! _dynamic_array_synapses_2_Apresyn_PC_I.empty() )
        {
            outfile__dynamic_array_synapses_2_Apresyn_PC_I.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_Apresyn_PC_I[0]), _dynamic_array_synapses_2_Apresyn_PC_I.size()*sizeof(_dynamic_array_synapses_2_Apresyn_PC_I[0]));
            outfile__dynamic_array_synapses_2_Apresyn_PC_I.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_Apresyn_PC_I." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_delay;
    outfile__dynamic_array_synapses_2_delay.open(results_dir + "_dynamic_array_synapses_2_delay_3163926887", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_delay.is_open())
    {
        if (! _dynamic_array_synapses_2_delay.empty() )
        {
            outfile__dynamic_array_synapses_2_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_delay[0]), _dynamic_array_synapses_2_delay.size()*sizeof(_dynamic_array_synapses_2_delay[0]));
            outfile__dynamic_array_synapses_2_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_delay_1;
    outfile__dynamic_array_synapses_2_delay_1.open(results_dir + "_dynamic_array_synapses_2_delay_1_3154022833", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_delay_1.is_open())
    {
        if (! _dynamic_array_synapses_2_delay_1.empty() )
        {
            outfile__dynamic_array_synapses_2_delay_1.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_delay_1[0]), _dynamic_array_synapses_2_delay_1.size()*sizeof(_dynamic_array_synapses_2_delay_1[0]));
            outfile__dynamic_array_synapses_2_delay_1.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_delay_1." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_lastupdate;
    outfile__dynamic_array_synapses_2_lastupdate.open(results_dir + "_dynamic_array_synapses_2_lastupdate_551200724", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_lastupdate.is_open())
    {
        if (! _dynamic_array_synapses_2_lastupdate.empty() )
        {
            outfile__dynamic_array_synapses_2_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_lastupdate[0]), _dynamic_array_synapses_2_lastupdate.size()*sizeof(_dynamic_array_synapses_2_lastupdate[0]));
            outfile__dynamic_array_synapses_2_lastupdate.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_lastupdate." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_N_incoming;
    outfile__dynamic_array_synapses_2_N_incoming.open(results_dir + "_dynamic_array_synapses_2_N_incoming_3109283082", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_incoming.is_open())
    {
        if (! _dynamic_array_synapses_2_N_incoming.empty() )
        {
            outfile__dynamic_array_synapses_2_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_N_incoming[0]), _dynamic_array_synapses_2_N_incoming.size()*sizeof(_dynamic_array_synapses_2_N_incoming[0]));
            outfile__dynamic_array_synapses_2_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_N_outgoing;
    outfile__dynamic_array_synapses_2_N_outgoing.open(results_dir + "_dynamic_array_synapses_2_N_outgoing_2656015824", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_outgoing.is_open())
    {
        if (! _dynamic_array_synapses_2_N_outgoing.empty() )
        {
            outfile__dynamic_array_synapses_2_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_N_outgoing[0]), _dynamic_array_synapses_2_N_outgoing.size()*sizeof(_dynamic_array_synapses_2_N_outgoing[0]));
            outfile__dynamic_array_synapses_2_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_w_PC_I;
    outfile__dynamic_array_synapses_2_w_PC_I.open(results_dir + "_dynamic_array_synapses_2_w_PC_I_575795538", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_w_PC_I.is_open())
    {
        if (! _dynamic_array_synapses_2_w_PC_I.empty() )
        {
            outfile__dynamic_array_synapses_2_w_PC_I.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_w_PC_I[0]), _dynamic_array_synapses_2_w_PC_I.size()*sizeof(_dynamic_array_synapses_2_w_PC_I[0]));
            outfile__dynamic_array_synapses_2_w_PC_I.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_w_PC_I." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3__synaptic_post;
    outfile__dynamic_array_synapses_3__synaptic_post.open(results_dir + "_dynamic_array_synapses_3__synaptic_post_4035665760", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3__synaptic_post.is_open())
    {
        if (! _dynamic_array_synapses_3__synaptic_post.empty() )
        {
            outfile__dynamic_array_synapses_3__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_3__synaptic_post[0]), _dynamic_array_synapses_3__synaptic_post.size()*sizeof(_dynamic_array_synapses_3__synaptic_post[0]));
            outfile__dynamic_array_synapses_3__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3__synaptic_pre;
    outfile__dynamic_array_synapses_3__synaptic_pre.open(results_dir + "_dynamic_array_synapses_3__synaptic_pre_2149485967", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3__synaptic_pre.is_open())
    {
        if (! _dynamic_array_synapses_3__synaptic_pre.empty() )
        {
            outfile__dynamic_array_synapses_3__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_3__synaptic_pre[0]), _dynamic_array_synapses_3__synaptic_pre.size()*sizeof(_dynamic_array_synapses_3__synaptic_pre[0]));
            outfile__dynamic_array_synapses_3__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3_Apostsyn_BC_E;
    outfile__dynamic_array_synapses_3_Apostsyn_BC_E.open(results_dir + "_dynamic_array_synapses_3_Apostsyn_BC_E_3266534219", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_Apostsyn_BC_E.is_open())
    {
        if (! _dynamic_array_synapses_3_Apostsyn_BC_E.empty() )
        {
            outfile__dynamic_array_synapses_3_Apostsyn_BC_E.write(reinterpret_cast<char*>(&_dynamic_array_synapses_3_Apostsyn_BC_E[0]), _dynamic_array_synapses_3_Apostsyn_BC_E.size()*sizeof(_dynamic_array_synapses_3_Apostsyn_BC_E[0]));
            outfile__dynamic_array_synapses_3_Apostsyn_BC_E.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_Apostsyn_BC_E." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3_Apresyn_BC_E;
    outfile__dynamic_array_synapses_3_Apresyn_BC_E.open(results_dir + "_dynamic_array_synapses_3_Apresyn_BC_E_4073134444", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_Apresyn_BC_E.is_open())
    {
        if (! _dynamic_array_synapses_3_Apresyn_BC_E.empty() )
        {
            outfile__dynamic_array_synapses_3_Apresyn_BC_E.write(reinterpret_cast<char*>(&_dynamic_array_synapses_3_Apresyn_BC_E[0]), _dynamic_array_synapses_3_Apresyn_BC_E.size()*sizeof(_dynamic_array_synapses_3_Apresyn_BC_E[0]));
            outfile__dynamic_array_synapses_3_Apresyn_BC_E.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_Apresyn_BC_E." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3_delay;
    outfile__dynamic_array_synapses_3_delay.open(results_dir + "_dynamic_array_synapses_3_delay_451066579", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_delay.is_open())
    {
        if (! _dynamic_array_synapses_3_delay.empty() )
        {
            outfile__dynamic_array_synapses_3_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_3_delay[0]), _dynamic_array_synapses_3_delay.size()*sizeof(_dynamic_array_synapses_3_delay[0]));
            outfile__dynamic_array_synapses_3_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3_delay_1;
    outfile__dynamic_array_synapses_3_delay_1.open(results_dir + "_dynamic_array_synapses_3_delay_1_2894431730", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_delay_1.is_open())
    {
        if (! _dynamic_array_synapses_3_delay_1.empty() )
        {
            outfile__dynamic_array_synapses_3_delay_1.write(reinterpret_cast<char*>(&_dynamic_array_synapses_3_delay_1[0]), _dynamic_array_synapses_3_delay_1.size()*sizeof(_dynamic_array_synapses_3_delay_1[0]));
            outfile__dynamic_array_synapses_3_delay_1.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_delay_1." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3_lastupdate;
    outfile__dynamic_array_synapses_3_lastupdate.open(results_dir + "_dynamic_array_synapses_3_lastupdate_3145722811", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_lastupdate.is_open())
    {
        if (! _dynamic_array_synapses_3_lastupdate.empty() )
        {
            outfile__dynamic_array_synapses_3_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_3_lastupdate[0]), _dynamic_array_synapses_3_lastupdate.size()*sizeof(_dynamic_array_synapses_3_lastupdate[0]));
            outfile__dynamic_array_synapses_3_lastupdate.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_lastupdate." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3_N_incoming;
    outfile__dynamic_array_synapses_3_N_incoming.open(results_dir + "_dynamic_array_synapses_3_N_incoming_586590565", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_N_incoming.is_open())
    {
        if (! _dynamic_array_synapses_3_N_incoming.empty() )
        {
            outfile__dynamic_array_synapses_3_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_3_N_incoming[0]), _dynamic_array_synapses_3_N_incoming.size()*sizeof(_dynamic_array_synapses_3_N_incoming[0]));
            outfile__dynamic_array_synapses_3_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3_N_outgoing;
    outfile__dynamic_array_synapses_3_N_outgoing.open(results_dir + "_dynamic_array_synapses_3_N_outgoing_99277247", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_N_outgoing.is_open())
    {
        if (! _dynamic_array_synapses_3_N_outgoing.empty() )
        {
            outfile__dynamic_array_synapses_3_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_3_N_outgoing[0]), _dynamic_array_synapses_3_N_outgoing.size()*sizeof(_dynamic_array_synapses_3_N_outgoing[0]));
            outfile__dynamic_array_synapses_3_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses_3_w_BC_E;
    outfile__dynamic_array_synapses_3_w_BC_E.open(results_dir + "_dynamic_array_synapses_3_w_BC_E_492643059", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_3_w_BC_E.is_open())
    {
        if (! _dynamic_array_synapses_3_w_BC_E.empty() )
        {
            outfile__dynamic_array_synapses_3_w_BC_E.write(reinterpret_cast<char*>(&_dynamic_array_synapses_3_w_BC_E[0]), _dynamic_array_synapses_3_w_BC_E.size()*sizeof(_dynamic_array_synapses_3_w_BC_E[0]));
            outfile__dynamic_array_synapses_3_w_BC_E.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_3_w_BC_E." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4__synaptic_post;
    outfile__dynamic_array_synapses_4__synaptic_post.open(results_dir + "_dynamic_array_synapses_4__synaptic_post_225617685", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4__synaptic_post.is_open())
    {
        if (! _dynamic_array_synapses_4__synaptic_post.empty() )
        {
            outfile__dynamic_array_synapses_4__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4__synaptic_post[0]), _dynamic_array_synapses_4__synaptic_post.size()*sizeof(_dynamic_array_synapses_4__synaptic_post[0]));
            outfile__dynamic_array_synapses_4__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4__synaptic_pre;
    outfile__dynamic_array_synapses_4__synaptic_pre.open(results_dir + "_dynamic_array_synapses_4__synaptic_pre_455049877", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4__synaptic_pre.is_open())
    {
        if (! _dynamic_array_synapses_4__synaptic_pre.empty() )
        {
            outfile__dynamic_array_synapses_4__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4__synaptic_pre[0]), _dynamic_array_synapses_4__synaptic_pre.size()*sizeof(_dynamic_array_synapses_4__synaptic_pre[0]));
            outfile__dynamic_array_synapses_4__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4_Apostsyn_BC_I;
    outfile__dynamic_array_synapses_4_Apostsyn_BC_I.open(results_dir + "_dynamic_array_synapses_4_Apostsyn_BC_I_1342445690", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_Apostsyn_BC_I.is_open())
    {
        if (! _dynamic_array_synapses_4_Apostsyn_BC_I.empty() )
        {
            outfile__dynamic_array_synapses_4_Apostsyn_BC_I.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_Apostsyn_BC_I[0]), _dynamic_array_synapses_4_Apostsyn_BC_I.size()*sizeof(_dynamic_array_synapses_4_Apostsyn_BC_I[0]));
            outfile__dynamic_array_synapses_4_Apostsyn_BC_I.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_Apostsyn_BC_I." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4_Apresyn_BC_I;
    outfile__dynamic_array_synapses_4_Apresyn_BC_I.open(results_dir + "_dynamic_array_synapses_4_Apresyn_BC_I_2663186311", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_Apresyn_BC_I.is_open())
    {
        if (! _dynamic_array_synapses_4_Apresyn_BC_I.empty() )
        {
            outfile__dynamic_array_synapses_4_Apresyn_BC_I.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_Apresyn_BC_I[0]), _dynamic_array_synapses_4_Apresyn_BC_I.size()*sizeof(_dynamic_array_synapses_4_Apresyn_BC_I[0]));
            outfile__dynamic_array_synapses_4_Apresyn_BC_I.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_Apresyn_BC_I." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4_delay;
    outfile__dynamic_array_synapses_4_delay.open(results_dir + "_dynamic_array_synapses_4_delay_3745875037", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_delay.is_open())
    {
        if (! _dynamic_array_synapses_4_delay.empty() )
        {
            outfile__dynamic_array_synapses_4_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_delay[0]), _dynamic_array_synapses_4_delay.size()*sizeof(_dynamic_array_synapses_4_delay[0]));
            outfile__dynamic_array_synapses_4_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4_delay_1;
    outfile__dynamic_array_synapses_4_delay_1.open(results_dir + "_dynamic_array_synapses_4_delay_1_3370444859", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_delay_1.is_open())
    {
        if (! _dynamic_array_synapses_4_delay_1.empty() )
        {
            outfile__dynamic_array_synapses_4_delay_1.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_delay_1[0]), _dynamic_array_synapses_4_delay_1.size()*sizeof(_dynamic_array_synapses_4_delay_1[0]));
            outfile__dynamic_array_synapses_4_delay_1.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_delay_1." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4_lastupdate;
    outfile__dynamic_array_synapses_4_lastupdate.open(results_dir + "_dynamic_array_synapses_4_lastupdate_3488023092", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_lastupdate.is_open())
    {
        if (! _dynamic_array_synapses_4_lastupdate.empty() )
        {
            outfile__dynamic_array_synapses_4_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_lastupdate[0]), _dynamic_array_synapses_4_lastupdate.size()*sizeof(_dynamic_array_synapses_4_lastupdate[0]));
            outfile__dynamic_array_synapses_4_lastupdate.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_lastupdate." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4_N_incoming;
    outfile__dynamic_array_synapses_4_N_incoming.open(results_dir + "_dynamic_array_synapses_4_N_incoming_1450066154", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_N_incoming.is_open())
    {
        if (! _dynamic_array_synapses_4_N_incoming.empty() )
        {
            outfile__dynamic_array_synapses_4_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_N_incoming[0]), _dynamic_array_synapses_4_N_incoming.size()*sizeof(_dynamic_array_synapses_4_N_incoming[0]));
            outfile__dynamic_array_synapses_4_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4_N_outgoing;
    outfile__dynamic_array_synapses_4_N_outgoing.open(results_dir + "_dynamic_array_synapses_4_N_outgoing_1903308848", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_N_outgoing.is_open())
    {
        if (! _dynamic_array_synapses_4_N_outgoing.empty() )
        {
            outfile__dynamic_array_synapses_4_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_N_outgoing[0]), _dynamic_array_synapses_4_N_outgoing.size()*sizeof(_dynamic_array_synapses_4_N_outgoing[0]));
            outfile__dynamic_array_synapses_4_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses_4_w_BC_I;
    outfile__dynamic_array_synapses_4_w_BC_I.open(results_dir + "_dynamic_array_synapses_4_w_BC_I_506357697", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_4_w_BC_I.is_open())
    {
        if (! _dynamic_array_synapses_4_w_BC_I.empty() )
        {
            outfile__dynamic_array_synapses_4_w_BC_I.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_w_BC_I[0]), _dynamic_array_synapses_4_w_BC_I.size()*sizeof(_dynamic_array_synapses_4_w_BC_I[0]));
            outfile__dynamic_array_synapses_4_w_BC_I.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_4_w_BC_I." << endl;
    }
    ofstream outfile__dynamic_array_synapses_5__synaptic_post;
    outfile__dynamic_array_synapses_5__synaptic_post.open(results_dir + "_dynamic_array_synapses_5__synaptic_post_2736404100", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5__synaptic_post.is_open())
    {
        if (! _dynamic_array_synapses_5__synaptic_post.empty() )
        {
            outfile__dynamic_array_synapses_5__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5__synaptic_post[0]), _dynamic_array_synapses_5__synaptic_post.size()*sizeof(_dynamic_array_synapses_5__synaptic_post[0]));
            outfile__dynamic_array_synapses_5__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_5__synaptic_pre;
    outfile__dynamic_array_synapses_5__synaptic_pre.open(results_dir + "_dynamic_array_synapses_5__synaptic_pre_2732874109", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5__synaptic_pre.is_open())
    {
        if (! _dynamic_array_synapses_5__synaptic_pre.empty() )
        {
            outfile__dynamic_array_synapses_5__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5__synaptic_pre[0]), _dynamic_array_synapses_5__synaptic_pre.size()*sizeof(_dynamic_array_synapses_5__synaptic_pre[0]));
            outfile__dynamic_array_synapses_5__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_5_delay;
    outfile__dynamic_array_synapses_5_delay.open(results_dir + "_dynamic_array_synapses_5_delay_2033356777", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5_delay.is_open())
    {
        if (! _dynamic_array_synapses_5_delay.empty() )
        {
            outfile__dynamic_array_synapses_5_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5_delay[0]), _dynamic_array_synapses_5_delay.size()*sizeof(_dynamic_array_synapses_5_delay[0]));
            outfile__dynamic_array_synapses_5_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_5_N_incoming;
    outfile__dynamic_array_synapses_5_N_incoming.open(results_dir + "_dynamic_array_synapses_5_N_incoming_3452636293", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5_N_incoming.is_open())
    {
        if (! _dynamic_array_synapses_5_N_incoming.empty() )
        {
            outfile__dynamic_array_synapses_5_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5_N_incoming[0]), _dynamic_array_synapses_5_N_incoming.size()*sizeof(_dynamic_array_synapses_5_N_incoming[0]));
            outfile__dynamic_array_synapses_5_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_synapses_5_N_outgoing;
    outfile__dynamic_array_synapses_5_N_outgoing.open(results_dir + "_dynamic_array_synapses_5_N_outgoing_3939990623", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_5_N_outgoing.is_open())
    {
        if (! _dynamic_array_synapses_5_N_outgoing.empty() )
        {
            outfile__dynamic_array_synapses_5_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5_N_outgoing[0]), _dynamic_array_synapses_5_N_outgoing.size()*sizeof(_dynamic_array_synapses_5_N_outgoing[0]));
            outfile__dynamic_array_synapses_5_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_5_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6__synaptic_post;
    outfile__dynamic_array_synapses_6__synaptic_post.open(results_dir + "_dynamic_array_synapses_6__synaptic_post_2329051766", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6__synaptic_post.is_open())
    {
        if (! _dynamic_array_synapses_6__synaptic_post.empty() )
        {
            outfile__dynamic_array_synapses_6__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6__synaptic_post[0]), _dynamic_array_synapses_6__synaptic_post.size()*sizeof(_dynamic_array_synapses_6__synaptic_post[0]));
            outfile__dynamic_array_synapses_6__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6__synaptic_pre;
    outfile__dynamic_array_synapses_6__synaptic_pre.open(results_dir + "_dynamic_array_synapses_6__synaptic_pre_3013161732", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6__synaptic_pre.is_open())
    {
        if (! _dynamic_array_synapses_6__synaptic_pre.empty() )
        {
            outfile__dynamic_array_synapses_6__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6__synaptic_pre[0]), _dynamic_array_synapses_6__synaptic_pre.size()*sizeof(_dynamic_array_synapses_6__synaptic_pre[0]));
            outfile__dynamic_array_synapses_6__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6_Apostsyn;
    outfile__dynamic_array_synapses_6_Apostsyn.open(results_dir + "_dynamic_array_synapses_6_Apostsyn_778548576", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_Apostsyn.is_open())
    {
        if (! _dynamic_array_synapses_6_Apostsyn.empty() )
        {
            outfile__dynamic_array_synapses_6_Apostsyn.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_Apostsyn[0]), _dynamic_array_synapses_6_Apostsyn.size()*sizeof(_dynamic_array_synapses_6_Apostsyn[0]));
            outfile__dynamic_array_synapses_6_Apostsyn.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_Apostsyn." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6_Apresyn;
    outfile__dynamic_array_synapses_6_Apresyn.open(results_dir + "_dynamic_array_synapses_6_Apresyn_1302573453", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_Apresyn.is_open())
    {
        if (! _dynamic_array_synapses_6_Apresyn.empty() )
        {
            outfile__dynamic_array_synapses_6_Apresyn.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_Apresyn[0]), _dynamic_array_synapses_6_Apresyn.size()*sizeof(_dynamic_array_synapses_6_Apresyn[0]));
            outfile__dynamic_array_synapses_6_Apresyn.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_Apresyn." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6_delay;
    outfile__dynamic_array_synapses_6_delay.open(results_dir + "_dynamic_array_synapses_6_delay_1222284660", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_delay.is_open())
    {
        if (! _dynamic_array_synapses_6_delay.empty() )
        {
            outfile__dynamic_array_synapses_6_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_delay[0]), _dynamic_array_synapses_6_delay.size()*sizeof(_dynamic_array_synapses_6_delay[0]));
            outfile__dynamic_array_synapses_6_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6_delay_1;
    outfile__dynamic_array_synapses_6_delay_1.open(results_dir + "_dynamic_array_synapses_6_delay_1_3859988669", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_delay_1.is_open())
    {
        if (! _dynamic_array_synapses_6_delay_1.empty() )
        {
            outfile__dynamic_array_synapses_6_delay_1.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_delay_1[0]), _dynamic_array_synapses_6_delay_1.size()*sizeof(_dynamic_array_synapses_6_delay_1[0]));
            outfile__dynamic_array_synapses_6_delay_1.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_delay_1." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6_lastupdate;
    outfile__dynamic_array_synapses_6_lastupdate.open(results_dir + "_dynamic_array_synapses_6_lastupdate_601660587", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_lastupdate.is_open())
    {
        if (! _dynamic_array_synapses_6_lastupdate.empty() )
        {
            outfile__dynamic_array_synapses_6_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_lastupdate[0]), _dynamic_array_synapses_6_lastupdate.size()*sizeof(_dynamic_array_synapses_6_lastupdate[0]));
            outfile__dynamic_array_synapses_6_lastupdate.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_lastupdate." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6_N_incoming;
    outfile__dynamic_array_synapses_6_N_incoming.open(results_dir + "_dynamic_array_synapses_6_N_incoming_3126189685", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_N_incoming.is_open())
    {
        if (! _dynamic_array_synapses_6_N_incoming.empty() )
        {
            outfile__dynamic_array_synapses_6_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_N_incoming[0]), _dynamic_array_synapses_6_N_incoming.size()*sizeof(_dynamic_array_synapses_6_N_incoming[0]));
            outfile__dynamic_array_synapses_6_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6_N_outgoing;
    outfile__dynamic_array_synapses_6_N_outgoing.open(results_dir + "_dynamic_array_synapses_6_N_outgoing_2638851759", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_N_outgoing.is_open())
    {
        if (! _dynamic_array_synapses_6_N_outgoing.empty() )
        {
            outfile__dynamic_array_synapses_6_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_N_outgoing[0]), _dynamic_array_synapses_6_N_outgoing.size()*sizeof(_dynamic_array_synapses_6_N_outgoing[0]));
            outfile__dynamic_array_synapses_6_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses_6_w_exc;
    outfile__dynamic_array_synapses_6_w_exc.open(results_dir + "_dynamic_array_synapses_6_w_exc_2579434254", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_6_w_exc.is_open())
    {
        if (! _dynamic_array_synapses_6_w_exc.empty() )
        {
            outfile__dynamic_array_synapses_6_w_exc.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_w_exc[0]), _dynamic_array_synapses_6_w_exc.size()*sizeof(_dynamic_array_synapses_6_w_exc[0]));
            outfile__dynamic_array_synapses_6_w_exc.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_6_w_exc." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_post;
    outfile__dynamic_array_synapses__synaptic_post.open(results_dir + "_dynamic_array_synapses__synaptic_post_1801389495", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_post.is_open())
    {
        if (! _dynamic_array_synapses__synaptic_post.empty() )
        {
            outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_post[0]), _dynamic_array_synapses__synaptic_post.size()*sizeof(_dynamic_array_synapses__synaptic_post[0]));
            outfile__dynamic_array_synapses__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_pre;
    outfile__dynamic_array_synapses__synaptic_pre.open(results_dir + "_dynamic_array_synapses__synaptic_pre_814148175", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
    {
        if (! _dynamic_array_synapses__synaptic_pre.empty() )
        {
            outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_pre[0]), _dynamic_array_synapses__synaptic_pre.size()*sizeof(_dynamic_array_synapses__synaptic_pre[0]));
            outfile__dynamic_array_synapses__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_delay;
    outfile__dynamic_array_synapses_delay.open(results_dir + "_dynamic_array_synapses_delay_3246960869", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_delay.is_open())
    {
        if (! _dynamic_array_synapses_delay.empty() )
        {
            outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_delay[0]), _dynamic_array_synapses_delay.size()*sizeof(_dynamic_array_synapses_delay[0]));
            outfile__dynamic_array_synapses_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_N_incoming;
    outfile__dynamic_array_synapses_N_incoming.open(results_dir + "_dynamic_array_synapses_N_incoming_1151751685", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_incoming.is_open())
    {
        if (! _dynamic_array_synapses_N_incoming.empty() )
        {
            outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_incoming[0]), _dynamic_array_synapses_N_incoming.size()*sizeof(_dynamic_array_synapses_N_incoming[0]));
            outfile__dynamic_array_synapses_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_synapses_N_outgoing;
    outfile__dynamic_array_synapses_N_outgoing.open(results_dir + "_dynamic_array_synapses_N_outgoing_1673144031", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_outgoing.is_open())
    {
        if (! _dynamic_array_synapses_N_outgoing.empty() )
        {
            outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_outgoing[0]), _dynamic_array_synapses_N_outgoing.size()*sizeof(_dynamic_array_synapses_N_outgoing[0]));
            outfile__dynamic_array_synapses_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
    }

    ofstream outfile__dynamic_array_statemonitor_w_exc;
    outfile__dynamic_array_statemonitor_w_exc.open(results_dir + "_dynamic_array_statemonitor_w_exc_3539290755", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_w_exc.is_open())
    {
        for (int n=0; n<_dynamic_array_statemonitor_w_exc.n; n++)
        {
            if (! _dynamic_array_statemonitor_w_exc(n).empty())
            {
                outfile__dynamic_array_statemonitor_w_exc.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_w_exc(n, 0)), _dynamic_array_statemonitor_w_exc.m*sizeof(_dynamic_array_statemonitor_w_exc(0, 0)));
            }
        }
        outfile__dynamic_array_statemonitor_w_exc.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_w_exc." << endl;
    }

    // Write spike queue states to disk
    ofstream outfile_synapses_pre;
    outfile_synapses_pre.open(results_dir + "synapses_pre_queue", ios::out);
    if (outfile_synapses_pre.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_pre << *synapses_pre.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_pre' for file" << std::endl;
    }
    ofstream outfile_synapses_1_post;
    outfile_synapses_1_post.open(results_dir + "synapses_1_post_queue", ios::out);
    if (outfile_synapses_1_post.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_1_post << *synapses_1_post.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_1_post' for file" << std::endl;
    }
    ofstream outfile_synapses_1_pre;
    outfile_synapses_1_pre.open(results_dir + "synapses_1_pre_queue", ios::out);
    if (outfile_synapses_1_pre.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_1_pre << *synapses_1_pre.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_1_pre' for file" << std::endl;
    }
    ofstream outfile_synapses_2_post;
    outfile_synapses_2_post.open(results_dir + "synapses_2_post_queue", ios::out);
    if (outfile_synapses_2_post.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_2_post << *synapses_2_post.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_2_post' for file" << std::endl;
    }
    ofstream outfile_synapses_2_pre;
    outfile_synapses_2_pre.open(results_dir + "synapses_2_pre_queue", ios::out);
    if (outfile_synapses_2_pre.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_2_pre << *synapses_2_pre.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_2_pre' for file" << std::endl;
    }
    ofstream outfile_synapses_3_post;
    outfile_synapses_3_post.open(results_dir + "synapses_3_post_queue", ios::out);
    if (outfile_synapses_3_post.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_3_post << *synapses_3_post.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_3_post' for file" << std::endl;
    }
    ofstream outfile_synapses_3_pre;
    outfile_synapses_3_pre.open(results_dir + "synapses_3_pre_queue", ios::out);
    if (outfile_synapses_3_pre.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_3_pre << *synapses_3_pre.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_3_pre' for file" << std::endl;
    }
    ofstream outfile_synapses_4_post;
    outfile_synapses_4_post.open(results_dir + "synapses_4_post_queue", ios::out);
    if (outfile_synapses_4_post.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_4_post << *synapses_4_post.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_4_post' for file" << std::endl;
    }
    ofstream outfile_synapses_4_pre;
    outfile_synapses_4_pre.open(results_dir + "synapses_4_pre_queue", ios::out);
    if (outfile_synapses_4_pre.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_4_pre << *synapses_4_pre.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_4_pre' for file" << std::endl;
    }
    ofstream outfile_synapses_5_pre;
    outfile_synapses_5_pre.open(results_dir + "synapses_5_pre_queue", ios::out);
    if (outfile_synapses_5_pre.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_5_pre << *synapses_5_pre.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_5_pre' for file" << std::endl;
    }
    ofstream outfile_synapses_6_post;
    outfile_synapses_6_post.open(results_dir + "synapses_6_post_queue", ios::out);
    if (outfile_synapses_6_post.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_6_post << *synapses_6_post.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_6_post' for file" << std::endl;
    }
    ofstream outfile_synapses_6_pre;
    outfile_synapses_6_pre.open(results_dir + "synapses_6_pre_queue", ios::out);
    if (outfile_synapses_6_pre.is_open()) {
        for (int i=0; i<1; i++) {
            outfile_synapses_6_pre << *synapses_6_pre.queue[i] << "\n";
        }
    } else {
        std::cout << "Error writing spike queue state for 'synapses_6_pre' for file" << std::endl;
    }

    // Write random generator state to disk
    ofstream random_generator_state;
    random_generator_state.open(results_dir + "random_generator_state", ios::out);
    if (random_generator_state.is_open()) {
        for (int i=0; i<1; i++)
            random_generator_state << _random_generators[i] << "\n";
    } else {
        std::cout << "Error writing random generator state to file." << std::endl;
    }

    // Write last run info to disk
    ofstream outfile_last_run_info;
    outfile_last_run_info.open(results_dir + "last_run_info.txt", ios::out);
    if(outfile_last_run_info.is_open())
    {
        outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
        outfile_last_run_info.close();
    } else
    {
        std::cout << "Error writing last run info to file." << std::endl;
    }
}

void _dealloc_arrays()
{
    using namespace brian;


    // static arrays
    if(_static_array__array_statemonitor__indices!=0)
    {
        delete [] _static_array__array_statemonitor__indices;
        _static_array__array_statemonitor__indices = 0;
    }
    if(_static_array__array_synapses_1_sources!=0)
    {
        delete [] _static_array__array_synapses_1_sources;
        _static_array__array_synapses_1_sources = 0;
    }
    if(_static_array__array_synapses_1_targets!=0)
    {
        delete [] _static_array__array_synapses_1_targets;
        _static_array__array_synapses_1_targets = 0;
    }
    if(_static_array__array_synapses_2_sources!=0)
    {
        delete [] _static_array__array_synapses_2_sources;
        _static_array__array_synapses_2_sources = 0;
    }
    if(_static_array__array_synapses_2_targets!=0)
    {
        delete [] _static_array__array_synapses_2_targets;
        _static_array__array_synapses_2_targets = 0;
    }
    if(_static_array__array_synapses_3_sources!=0)
    {
        delete [] _static_array__array_synapses_3_sources;
        _static_array__array_synapses_3_sources = 0;
    }
    if(_static_array__array_synapses_3_targets!=0)
    {
        delete [] _static_array__array_synapses_3_targets;
        _static_array__array_synapses_3_targets = 0;
    }
    if(_static_array__array_synapses_4_sources!=0)
    {
        delete [] _static_array__array_synapses_4_sources;
        _static_array__array_synapses_4_sources = 0;
    }
    if(_static_array__array_synapses_4_targets!=0)
    {
        delete [] _static_array__array_synapses_4_targets;
        _static_array__array_synapses_4_targets = 0;
    }
    if(_static_array__array_synapses_6_sources!=0)
    {
        delete [] _static_array__array_synapses_6_sources;
        _static_array__array_synapses_6_sources = 0;
    }
    if(_static_array__array_synapses_6_targets!=0)
    {
        delete [] _static_array__array_synapses_6_targets;
        _static_array__array_synapses_6_targets = 0;
    }
    if(_static_array__dynamic_array_synapses_1_w_exc!=0)
    {
        delete [] _static_array__dynamic_array_synapses_1_w_exc;
        _static_array__dynamic_array_synapses_1_w_exc = 0;
    }
    if(_static_array__dynamic_array_synapses_2_w_PC_I!=0)
    {
        delete [] _static_array__dynamic_array_synapses_2_w_PC_I;
        _static_array__dynamic_array_synapses_2_w_PC_I = 0;
    }
    if(_static_array__dynamic_array_synapses_3_w_BC_E!=0)
    {
        delete [] _static_array__dynamic_array_synapses_3_w_BC_E;
        _static_array__dynamic_array_synapses_3_w_BC_E = 0;
    }
    if(_static_array__dynamic_array_synapses_4_w_BC_I!=0)
    {
        delete [] _static_array__dynamic_array_synapses_4_w_BC_I;
        _static_array__dynamic_array_synapses_4_w_BC_I = 0;
    }
    if(_static_array__dynamic_array_synapses_6_w_exc!=0)
    {
        delete [] _static_array__dynamic_array_synapses_6_w_exc;
        _static_array__dynamic_array_synapses_6_w_exc = 0;
    }
}

