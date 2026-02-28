#include "code_objects/synapses_1_post_codeobject_1.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<chrono>
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>
#include "brianlib/stdint_compat.h"
#include "synapses_classes.h"

////// SUPPORT CODE ///////
namespace {
        
    template <typename T>
    static inline T _clip(const T value, const double a_min, const double a_max)
    {
        if (value < a_min)
            return a_min;
        if (value > a_max)
            return a_max;
        return value;
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



void _run_synapses_1_post_codeobject_1()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    double* const _array_synapses_1_Apostsyn = _dynamic_array_synapses_1_Apostsyn.empty()? 0 : &_dynamic_array_synapses_1_Apostsyn[0];
const size_t _numApostsyn = _dynamic_array_synapses_1_Apostsyn.size();
double* const _array_synapses_1_Apresyn = _dynamic_array_synapses_1_Apresyn.empty()? 0 : &_dynamic_array_synapses_1_Apresyn[0];
const size_t _numApresyn = _dynamic_array_synapses_1_Apresyn.size();
int32_t* const _array_synapses_1__synaptic_pre = _dynamic_array_synapses_1__synaptic_pre.empty()? 0 : &_dynamic_array_synapses_1__synaptic_pre[0];
const size_t _num_synaptic_pre = _dynamic_array_synapses_1__synaptic_pre.size();
const double dApostsyn = 0.018433841866281598;
double* const _array_synapses_1_lastupdate = _dynamic_array_synapses_1_lastupdate.empty()? 0 : &_dynamic_array_synapses_1_lastupdate[0];
const size_t _numlastupdate = _dynamic_array_synapses_1_lastupdate.size();
const size_t _numt = 1;
const double taum = 0.014621020600447083;
const double taup = 0.014621020600447083;
double* const _array_synapses_1_w_exc = _dynamic_array_synapses_1_w_exc.empty()? 0 : &_dynamic_array_synapses_1_w_exc[0];
const size_t _numw_exc = _dynamic_array_synapses_1_w_exc.size();
const double wmax = 6.0;
    ///// POINTERS ////////////
        
    double* __restrict  _ptr_array_synapses_1_Apostsyn = _array_synapses_1_Apostsyn;
    double* __restrict  _ptr_array_synapses_1_Apresyn = _array_synapses_1_Apresyn;
    int32_t* __restrict  _ptr_array_synapses_1__synaptic_pre = _array_synapses_1__synaptic_pre;
    double* __restrict  _ptr_array_synapses_1_lastupdate = _array_synapses_1_lastupdate;
    double*   _ptr_array_defaultclock_t = _array_defaultclock_t;
    double* __restrict  _ptr_array_synapses_1_w_exc = _array_synapses_1_w_exc;



    // This is only needed for the _debugmsg function below

    // scalar code
    const size_t _vectorisation_idx = -1;
        
    const double t = _ptr_array_defaultclock_t[0];
    const double _lio_1 = 1.0f*1.0/taum;
    const double _lio_2 = 1.0f*1.0/taup;


    
    {
    std::vector<int> *_spiking_synapses = synapses_1_post.peek();
    const int _num_spiking_synapses = _spiking_synapses->size();

    
    for(int _spiking_synapse_idx=0;
        _spiking_synapse_idx<_num_spiking_synapses;
        _spiking_synapse_idx++)
    {
        const size_t _idx = (*_spiking_synapses)[_spiking_synapse_idx];
        const size_t _vectorisation_idx = _idx;
                
        double Apostsyn = _ptr_array_synapses_1_Apostsyn[_idx];
        double Apresyn = _ptr_array_synapses_1_Apresyn[_idx];
        double lastupdate = _ptr_array_synapses_1_lastupdate[_idx];
        double w_exc = _ptr_array_synapses_1_w_exc[_idx];
        const double _Apostsyn = Apostsyn * exp(_lio_1 * (- (t - lastupdate)));
        const double _Apresyn = Apresyn * exp(_lio_2 * (- (t - lastupdate)));
        Apostsyn = _Apostsyn;
        Apresyn = _Apresyn;
        Apostsyn += dApostsyn;
        w_exc = _clip(w_exc + Apresyn, 0, wmax);
        lastupdate = t;
        _ptr_array_synapses_1_Apostsyn[_idx] = Apostsyn;
        _ptr_array_synapses_1_Apresyn[_idx] = Apresyn;
        _ptr_array_synapses_1_lastupdate[_idx] = lastupdate;
        _ptr_array_synapses_1_w_exc[_idx] = w_exc;

    }

    }

}

void _debugmsg_synapses_1_post_codeobject_1()
{
    using namespace brian;
    std::cout << "Number of synapses: " << _dynamic_array_synapses_1__synaptic_pre.size() << endl;
}

