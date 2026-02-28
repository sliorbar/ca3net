#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<chrono>
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>

////// SUPPORT CODE ///////
namespace {
        
    static inline int64_t _timestep(double t, double dt)
    {
        return (int64_t)((t + 1e-3*dt)/dt);
    }
    template < typename T1, typename T2 > struct _higher_type;
    template < > struct _higher_type<int32_t,int32_t> { typedef int32_t type; };
    template < > struct _higher_type<int32_t,int64_t> { typedef int64_t type; };
    template < > struct _higher_type<int32_t,float> { typedef float type; };
    template < > struct _higher_type<int32_t,double> { typedef double type; };
    template < > struct _higher_type<int32_t,long double> { typedef long double type; };
    template < > struct _higher_type<int64_t,int32_t> { typedef int64_t type; };
    template < > struct _higher_type<int64_t,int64_t> { typedef int64_t type; };
    template < > struct _higher_type<int64_t,float> { typedef float type; };
    template < > struct _higher_type<int64_t,double> { typedef double type; };
    template < > struct _higher_type<int64_t,long double> { typedef long double type; };
    template < > struct _higher_type<float,int32_t> { typedef float type; };
    template < > struct _higher_type<float,int64_t> { typedef float type; };
    template < > struct _higher_type<float,float> { typedef float type; };
    template < > struct _higher_type<float,double> { typedef double type; };
    template < > struct _higher_type<float,long double> { typedef long double type; };
    template < > struct _higher_type<double,int32_t> { typedef double type; };
    template < > struct _higher_type<double,int64_t> { typedef double type; };
    template < > struct _higher_type<double,float> { typedef double type; };
    template < > struct _higher_type<double,double> { typedef double type; };
    template < > struct _higher_type<double,long double> { typedef long double type; };
    template < > struct _higher_type<long double,int32_t> { typedef long double type; };
    template < > struct _higher_type<long double,int64_t> { typedef long double type; };
    template < > struct _higher_type<long double,float> { typedef long double type; };
    template < > struct _higher_type<long double,double> { typedef long double type; };
    template < > struct _higher_type<long double,long double> { typedef long double type; };
    // General template, used for floating point types
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_mod(T1 x, T2 y)
    {
        return x-y*floor(1.0*x/y);
    }
    // Specific implementations for integer types
    // (from Cython, see LICENSE file)
    template <>
    inline int32_t _brian_mod(int32_t x, int32_t y)
    {
        int32_t r = x % y;
        r += ((r != 0) & ((r ^ y) < 0)) * y;
        return r;
    }
    template <>
    inline int64_t _brian_mod(int32_t x, int64_t y)
    {
        int64_t r = x % y;
        r += ((r != 0) & ((r ^ y) < 0)) * y;
        return r;
    }
    template <>
    inline int64_t _brian_mod(int64_t x, int32_t y)
    {
        int64_t r = x % y;
        r += ((r != 0) & ((r ^ y) < 0)) * y;
        return r;
    }
    template <>
    inline int64_t _brian_mod(int64_t x, int64_t y)
    {
        int64_t r = x % y;
        r += ((r != 0) & ((r ^ y) < 0)) * y;
        return r;
    }
    // General implementation, used for floating point types
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_floordiv(T1 x, T2 y)
    {{
        return floor(1.0*x/y);
    }}
    // Specific implementations for integer types
    // (from Cython, see LICENSE file)
    template <>
    inline int32_t _brian_floordiv<int32_t, int32_t>(int32_t a, int32_t b) {
        int32_t q = a / b;
        int32_t r = a - q*b;
        q -= ((r != 0) & ((r ^ b) < 0));
        return q;
    }
    template <>
    inline int64_t _brian_floordiv<int32_t, int64_t>(int32_t a, int64_t b) {
        int64_t q = a / b;
        int64_t r = a - q*b;
        q -= ((r != 0) & ((r ^ b) < 0));
        return q;
    }
    template <>
    inline int64_t _brian_floordiv<int64_t, int>(int64_t a, int32_t b) {
        int64_t q = a / b;
        int64_t r = a - q*b;
        q -= ((r != 0) & ((r ^ b) < 0));
        return q;
    }
    template <>
    inline int64_t _brian_floordiv<int64_t, int64_t>(int64_t a, int64_t b) {
        int64_t q = a / b;
        int64_t r = a - q*b;
        q -= ((r != 0) & ((r ^ b) < 0));
        return q;
    }
    #ifdef _MSC_VER
    #define _brian_pow(x, y) (pow((double)(x), (y)))
    #else
    #define _brian_pow(x, y) (pow((x), (y)))
    #endif

}

////// HASH DEFINES ///////



void _run_neurongroup_stateupdater_codeobject()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    const double Cm_PC = 1.8013636547792772e-10;
const double Erev_E = 0.0;
const double Erev_I = - 0.07;
const int64_t N = 8000;
const double Vrest_PC = - 0.0751884554193901;
const double a_PC = - 2.74347065652738e-10;
const double decay_PC_E = 0.0095;
const double decay_PC_I = 0.0033;
const double decay_PC_MF = 0.0054;
const double delta_T_PC = 0.0042340696257631;
const size_t _numdt = 1;
const size_t _numg_ampa = 8000;
const size_t _numg_ampaMF = 8000;
const size_t _numg_gaba = 8000;
const double g_leak_PC = 4.314757919372231e-09;
const size_t _numlastspike = 8000;
const size_t _numnot_refractory = 8000;
const double rise_PC_E = 0.0013000000000000002;
const double rise_PC_I = 0.0003;
const double rise_PC_MF = 0.0006500000000000001;
const size_t _numt = 1;
const double tau_w_PC = 0.0849358017225512;
const double theta_PC = - 0.0244255910105977;
const size_t _numvm = 8000;
const size_t _numw = 8000;
const size_t _numx_ampa = 8000;
const size_t _numx_ampaMF = 8000;
const size_t _numx_gaba = 8000;
const double z = 1e-09;
    ///// POINTERS ////////////
        
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
    double* __restrict  _ptr_array_neurongroup_g_ampa = _array_neurongroup_g_ampa;
    double* __restrict  _ptr_array_neurongroup_g_ampaMF = _array_neurongroup_g_ampaMF;
    double* __restrict  _ptr_array_neurongroup_g_gaba = _array_neurongroup_g_gaba;
    double* __restrict  _ptr_array_neurongroup_lastspike = _array_neurongroup_lastspike;
    char* __restrict  _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;
    double*   _ptr_array_defaultclock_t = _array_defaultclock_t;
    double* __restrict  _ptr_array_neurongroup_vm = _array_neurongroup_vm;
    double* __restrict  _ptr_array_neurongroup_w = _array_neurongroup_w;
    double* __restrict  _ptr_array_neurongroup_x_ampa = _array_neurongroup_x_ampa;
    double* __restrict  _ptr_array_neurongroup_x_ampaMF = _array_neurongroup_x_ampaMF;
    double* __restrict  _ptr_array_neurongroup_x_gaba = _array_neurongroup_x_gaba;


    //// MAIN CODE ////////////
    // scalar code
    const size_t _vectorisation_idx = -1;
        
    const double dt = _ptr_array_defaultclock_dt[0];
    const double t = _ptr_array_defaultclock_t[0];
    const int64_t _lio_1 = _timestep(0.0059632693094559906, dt);
    const double _lio_2 = exp(1.0f*(- dt)/rise_PC_E);
    const double _lio_3 = exp(1.0f*(- dt)/rise_PC_MF);
    const double _lio_4 = exp(1.0f*(- dt)/rise_PC_I);
    const double _lio_5 = 1.0f*(Erev_E * z)/Cm_PC;
    const double _lio_6 = 1.0f*(Erev_I * z)/Cm_PC;
    const double _lio_7 = 1.0f*(Vrest_PC * g_leak_PC)/Cm_PC;
    const double _lio_8 = 1.0f*((delta_T_PC * g_leak_PC) * exp(1.0f*(- theta_PC)/delta_T_PC))/Cm_PC;
    const double _lio_9 = 1.0f*1.0/delta_T_PC;
    const double _lio_10 = 1.0f*1.0/Cm_PC;
    const double _lio_11 = 1.0f*z/Cm_PC;
    const double _lio_12 = 1.0f*g_leak_PC/Cm_PC;
    const double _lio_13 = 0.0 - _lio_12;
    const double _lio_14 = - tau_w_PC;
    const double _lio_15 = 1.0f*((- Vrest_PC) * a_PC)/tau_w_PC;
    const double _lio_16 = 1.0f*a_PC/tau_w_PC;
    const double _lio_17 = exp(1.0f*(- dt)/tau_w_PC);
    const double _lio_18 = exp(1.0f*(- dt)/decay_PC_E);
    const double _lio_19 = exp(1.0f*(- dt)/decay_PC_MF);
    const double _lio_20 = exp(1.0f*(- dt)/decay_PC_I);


    const int _N = N;
    
    for(int _idx=0; _idx<_N; _idx++)
    {
        // vector code
        const size_t _vectorisation_idx = _idx;
                
        double g_ampa = _ptr_array_neurongroup_g_ampa[_idx];
        double g_ampaMF = _ptr_array_neurongroup_g_ampaMF[_idx];
        double g_gaba = _ptr_array_neurongroup_g_gaba[_idx];
        const double lastspike = _ptr_array_neurongroup_lastspike[_idx];
        char not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
        double vm = _ptr_array_neurongroup_vm[_idx];
        double w = _ptr_array_neurongroup_w[_idx];
        double x_ampa = _ptr_array_neurongroup_x_ampa[_idx];
        double x_ampaMF = _ptr_array_neurongroup_x_ampaMF[_idx];
        double x_gaba = _ptr_array_neurongroup_x_gaba[_idx];
        not_refractory = _timestep(t - lastspike, dt) >= _lio_1;
        const double _BA_g_ampa = - x_ampa;
        const double _g_ampa = (- _BA_g_ampa) + (_lio_2 * (_BA_g_ampa + g_ampa));
        const double _BA_g_ampaMF = - x_ampaMF;
        const double _g_ampaMF = (- _BA_g_ampaMF) + (_lio_3 * (_BA_g_ampaMF + g_ampaMF));
        const double _BA_g_gaba = - x_gaba;
        const double _g_gaba = (- _BA_g_gaba) + (_lio_4 * (_BA_g_gaba + g_gaba));
        double _BA_vm;
        if(!not_refractory)
            _BA_vm = 0.0;
        else 
            _BA_vm = 1.0f*((_lio_7 + ((((_lio_5 * g_ampa) + (_lio_5 * g_ampaMF)) + (_lio_6 * g_gaba)) + (_lio_8 * exp(_lio_9 * vm)))) - (_lio_10 * w))/((_lio_13 + (_lio_11 * (- g_ampa))) - ((_lio_11 * g_ampaMF) + (_lio_11 * g_gaba)));
        double _vm;
        if(!not_refractory)
            _vm = (- _BA_vm) + (_BA_vm + vm);
        else 
            _vm = (- _BA_vm) + ((_BA_vm + vm) * exp(dt * ((_lio_13 + (_lio_11 * (- g_ampa))) - ((_lio_11 * g_ampaMF) + (_lio_11 * g_gaba)))));
        const double _BA_w = _lio_14 * (_lio_15 + (_lio_16 * vm));
        const double _w = (- _BA_w) + (_lio_17 * (_BA_w + w));
        const double _x_ampa = _lio_18 * x_ampa;
        const double _x_ampaMF = _lio_19 * x_ampaMF;
        const double _x_gaba = _lio_20 * x_gaba;
        g_ampa = _g_ampa;
        g_ampaMF = _g_ampaMF;
        g_gaba = _g_gaba;
        if(not_refractory)
            vm = _vm;
        w = _w;
        x_ampa = _x_ampa;
        x_ampaMF = _x_ampaMF;
        x_gaba = _x_gaba;
        _ptr_array_neurongroup_g_ampa[_idx] = g_ampa;
        _ptr_array_neurongroup_g_ampaMF[_idx] = g_ampaMF;
        _ptr_array_neurongroup_g_gaba[_idx] = g_gaba;
        _ptr_array_neurongroup_not_refractory[_idx] = not_refractory;
        _ptr_array_neurongroup_vm[_idx] = vm;
        _ptr_array_neurongroup_w[_idx] = w;
        _ptr_array_neurongroup_x_ampa[_idx] = x_ampa;
        _ptr_array_neurongroup_x_ampaMF[_idx] = x_ampaMF;
        _ptr_array_neurongroup_x_gaba[_idx] = x_gaba;

    }

}


