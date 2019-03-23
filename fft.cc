#include "complex.h"
#include <vector>
#include <complex>
#include <stddef.h>
#include <utility> // std::swap in c++11
#include <assert.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

template<typename T>
void dump_vector(const std::vector<T> & vec){
    for(const T & elem : vec){
        std::cout<<elem<<", ";
    }
    std::cout<<std::endl;
}

#ifndef ABS
#define ABS(x) ((x)>0?(x):-1*(x))
#endif

template<typename T>
int valid_vector(const std::vector<complex_t<T>> & lhs, const std::vector<complex_t<T>> & rhs, T delta = (T)0.001){
    assert(lhs.size() == rhs.size());
    size_t i;
    int err_cnt = 0;
    for(i = 0;i < lhs.size(); i++){
        T d_re = std::real(lhs[i]) - std::real(rhs[i]);
        T d_im = std::imag(lhs[i]) - std::imag(rhs[i]);
        d_re = ABS(d_re);
        d_im = ABS(d_im);
        if(d_re > delta || d_im > delta){
            std::cout<<" diff at "<<i<<", lhs:"<<lhs[i]<<", rhs:"<<rhs[i]<<std::endl;
            err_cnt++;
        }
    }
    return err_cnt;
}

template<typename T>
void rand_vec(std::vector<complex_t<T>> & seq, size_t len){
    static std::random_device rd;   // seed

    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(-2.0, 2.0);

    seq.resize(len);
    size_t i;
    for(i=0;i<seq.size();i++){
        seq[i] = complex_t<T>(dist(mt), dist(mt));
    }
}

// https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html
template<typename T>
void fft_naive(const std::vector<complex_t<T>> & t_seq, std::vector<complex_t<T>> & f_seq, size_t length=0){
    // https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Definition
    auto omega_func = [](size_t total_n, size_t k){
        // e^( -1 * 2PI*k*n/N * i), here n is iter through each 
        T r = (T)1;
        T theta = -1 * C_2PI*k / total_n;
        return std::polar2<T>(r, theta);
    };
    if(length == 0)
        length = t_seq.size();
    size_t fft_n = length;
    size_t k;
    std::vector<complex_t<T>> seq = t_seq;
    if(length > t_seq.size()){
        for(size_t i=0; i< (length - t_seq.size()); i++ ){
            seq.emplace_back(0,0);
        }
    }
    for(k=0;k<fft_n;k++){
        size_t n;
        complex_t<T> omega_k = omega_func(fft_n, k);
        complex_t<T> A_k;
        for(n=0;n<fft_n;n++){
            A_k +=  std::pow(omega_k, (T)n) * seq[n] ;
        }
        f_seq.push_back(A_k);
    }
}

template<typename T>
void ifft_naive(const std::vector<complex_t<T>> & f_seq, std::vector<complex_t<T>> & t_seq, size_t length=0){
    auto omega_func_inverse = [](size_t total_n, size_t k){
        // e^( 2PI*k*n/N * i), here n is iter through each 
        T r = (T)1;
        T theta = C_2PI*k / total_n;
        return std::polar2<T>(r, theta);
    };
    if(length == 0)
        length = f_seq.size();
    size_t fft_n = length;
    size_t k;

    std::vector<complex_t<T>> seq = f_seq;
    if(length > f_seq.size()){
        for(size_t i=0; i< (length - f_seq.size()); i++ ){
            seq.push_back(complex_t<T>());
        }
    }
    for(k=0;k<fft_n;k++){
        size_t n;
        complex_t<T> omega_k_inverse = omega_func_inverse(fft_n, k);
        complex_t<T> a_k;
        for(n=0;n<fft_n;n++){
            a_k +=  std::pow(omega_k_inverse, (T)n) * seq[n] ;
        }
        a_k /= (T)fft_n;
        t_seq.push_back(a_k);
    }
}

// https://en.wikipedia.org/wiki/Bit-reversal_permutation
// below function produce  https://oeis.org/A030109
void bit_reverse_permute(size_t radix2_num, std::vector<size_t> &arr)
{
    size_t k;
    arr.resize(std::pow(2,radix2_num));
    arr[0] = 0;
    for(k=0;k<radix2_num;k++){
       size_t last_k_len = std::pow(2, k);
       size_t last_k;
       for(last_k = 0; last_k < last_k_len; last_k++){
           arr[last_k] = 2*arr[last_k];
           arr[last_k_len+last_k] = arr[last_k]+1;
       }
    }
}

template<typename ELEMENT_T>
void bit_reverse_radix2(std::vector<ELEMENT_T> & vec){
    size_t length = vec.size();
    assert( ( (length & (length - 1)) == 0 ) && "must be radix of 2");
    std::vector<size_t> r_idx;
    bit_reverse_permute(std::log2(length), r_idx);
    size_t i;
    size_t ir;
    for(i=0;i<length;i++){
        ir = r_idx[i];
        //std::cout<<"i:"<<i<<", ir:"<<ir<<std::endl;
        if(i<ir){
            std::swap(vec[i], vec[ir]);
        }
    }
}

template<typename T>
void fft_cooley_tukey(const std::vector<complex_t<T>> & t_seq, std::vector<complex_t<T>> & f_seq, size_t length=0)
{
    auto omega_func = [](size_t total_n, size_t k){
        // e^( -1 * 2PI*k*n/N * i), here n is iter through each 
        T r = (T)1;
        T theta = -1 * C_2PI*k / total_n;
        return std::polar2<T>(r, theta);
    };


    // TODO, length or t_seq.size() is radix of 2
    if(length == 0)
        length = t_seq.size();
    size_t itr;

    if(length == 1){
        f_seq.emplace_back(std::real(t_seq[0]), std::imag(t_seq[0]));
        return ;
    }

    //http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    assert( ( (length & (length - 1)) == 0 ) && "current only length power of 2");

    for(const auto & s : t_seq){
        f_seq.emplace_back(s);
    }
    if(length > t_seq.size()){
        for(itr = 0; itr < (length-t_seq.size()); itr++)
            f_seq.emplace_back((T)0, (T)0);
    }
    bit_reverse_radix2(f_seq);

    /*
    * Wn^k -> omega_func(n,k)
    * 
    * W2^0 | W2^1
    * W4^0 W4^1 | W4^2 W4^3
    * W8^0 W8^1 W8^2 W8^3 | W8^4 W8^5 W8^6 W8^7
    * W16^0 W16^1 W16^2 W16^3 W16^4 W16^5 W16^6 W16^7 | ...
    * 
    */
    std::vector<complex_t<T>> omega_list;   // pre-compute omega, and index to it later
    omega_list.resize(length/2);
    for(itr = 0; itr < length/2 ; itr ++){
        omega_list[itr] = omega_func( length, itr);
    }

    // TODO: length == 1 case
    for(itr = 2; itr<=length; itr*=2){
        size_t group = length / itr;    // butterfly groups
        size_t g;
        for(g=0;g<group;g++){
            // group length is itr, have itr/2 even, and itr/2 odd
            size_t k_itr;
            for(k_itr = 0;k_itr < itr/2; k_itr++){
                size_t k = k_itr + g*itr;
                //auto omega_k = omega_func( itr , k_itr);
                auto & omega_k = omega_list[length/itr * k_itr];
                // b(k)     = a(k) + omega_k*a(k+n/2)    X(k) -> odd, X(k+n/2) -> even
                // b(k+n/2) = a(k) - omega_k*a(k+n/2)
                auto t = omega_k * f_seq[k+itr/2];
                f_seq[k+itr/2] = f_seq[k] - t;
                f_seq[k] += t;
            }
        }
    }
}
template<typename T>
void ifft_cooley_tukey(const std::vector<complex_t<T>> & f_seq, std::vector<complex_t<T>> & t_seq, size_t length=0){
    auto omega_func_inverse = [](size_t total_n, size_t k){
        // e^(  2PI*k*n/N * i), here n is iter through each 
        T r = (T)1;
        T theta = C_2PI*k / total_n;
        return std::polar2<T>(r, theta);
    };

    // TODO, length or f_seq.size() is radix of 2
    if(length == 0)
        length = f_seq.size();
    size_t itr;

    if(length == 1){
        t_seq.emplace_back(std::real(f_seq[0]), std::imag(f_seq[0]));
        return ;
    }

    //http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    assert( ( (length & (length - 1)) == 0 ) && "current only length power of 2");

    //for(const T & s : f_seq){
    //    t_seq.emplace_back(s, (T)0);
    //}
    t_seq = f_seq;
    if(length > t_seq.size()){
        for(itr = 0;itr < (length-t_seq.size()); itr++)
            t_seq.emplace_back((T)0, (T)0);
    }
    bit_reverse_radix2(t_seq);

    /*
    * Wn^k -> omega_func(n,k)
    * 
    * W2^0 | W2^1
    * W4^0 W4^1 | W4^2 W4^3
    * W8^0 W8^1 W8^2 W8^3 | W8^4 W8^5 W8^6 W8^7
    * W16^0 W16^1 W16^2 W16^3 W16^4 W16^5 W16^6 W16^7 | ...
    * 
    */
    std::vector<complex_t<T>> omega_list;   // pre-compute omega, and index to it later
    omega_list.resize(length/2);
    for(itr = 0; itr < length/2 ; itr ++){
        omega_list[itr] = omega_func_inverse( length, itr);
    }

    // TODO: length == 1 case
    for(itr = 2; itr<=length; itr*=2){
        size_t group = length / itr;    // butterfly groups
        size_t g;
        for(g=0;g<group;g++){
            // group length is itr, have itr/2 even, and itr/2 odd
            size_t k_itr;
            for(k_itr = 0;k_itr < itr/2; k_itr++){
                size_t k = k_itr + g*itr;
                //auto omega_k = omega_func( itr , k_itr);
                auto & omega_k = omega_list[length/itr * k_itr];
                // b(k)     = a(k) + omega_k*a(k+n/2)    X(k) -> odd, X(k+n/2) -> even
                // b(k+n/2) = a(k) - omega_k*a(k+n/2)
                auto t = omega_k * t_seq[k+itr/2];
                t_seq[k+itr/2] = t_seq[k] - t;
                t_seq[k] += t;
            }
        }
    }

    // inverse only, need devide
    for(itr = 0; itr < t_seq.size(); itr++)
        t_seq[itr] /= (T)length;
}

typedef double d_type;


template<typename T>
void convolve_naive(const std::vector<T> & data, const std::vector<T> & filter, std::vector<T> &  dst, bool correlation = false){
    std::vector<T> f = filter;
    std::vector<T> d = data;
    

    size_t dst_len = data.size() + filter.size() - 1;
    size_t pad     = filter.size()-1;
    size_t i, j;

    if(!correlation)
        std::reverse(f.begin(), f.end());
    d.reserve(data.size() + 2 * pad);
    d.insert(d.begin(), (T)0);
    d.insert(d.begin(), (T)0);
    d.push_back((T)0);
    d.push_back((T)0);
    
    dst.reserve(dst_len);

    for(i=0;i<dst_len;i++){
        T v = 0;
        for(j=0;j<filter.size();j++){
            v += f[j] * d[i+j];
        }
        dst.push_back(v);
    }
}

/*
* conv(a, b) = ifft(fft(a_and_zeros) * fft(b_and_zeros))
*
* corr(a, b) = ifft(fft(a_and_zeros) * conj(fft(b_and_zeros))) [1]
*  or
* corr(a, b) = ifft(fft(a_and_zeros) * fft(b_and_zeros[reversed]))
*
*
* [1]: www.claysturner.com/dsp/timereversal.pdf
*    indeed, for DFT, time reverse is not equal to conj in freq. a tiwddle factor is needed
*       f[n] -> F[k]
*       f[N-n-1] -> conj(F[k]) * e^(i*2PI*k/N)
*/

#define USE_CORR_CONJ
//#define USE_CORR_WARP_SHIFT
template<typename T>
void convolve_fft(const std::vector<T> & data, const std::vector<T> & filter, std::vector<T> &  dst, bool correlation = false){
    size_t dst_len = data.size()+filter.size()-1;

    // round to nearest 2^power number
    size_t fft_len = (size_t)std::pow(2, std::ceil(std::log2(dst_len)));
    std::vector<complex_t<T>> t_data;
    std::vector<complex_t<T>> t_filter;
    std::vector<complex_t<T>> f_data;
    std::vector<complex_t<T>> f_filter;
    std::vector<complex_t<T>> f_result;

    for(const auto & it :data)
        t_data.emplace_back(it , (T)0);
    for(const auto & it :filter)
        t_filter.emplace_back(it , (T)0);

    t_data.reserve(fft_len);
    t_filter.reserve(fft_len);

    // zero padding
    for(size_t i=0;i<(fft_len-t_data.size());i++){
        t_data.emplace_back((T)0, (T)0);
    }
#if defined( USE_CORR_CONJ )
    auto omega_func = [](size_t k, size_t N){
        T r = (T)1;
        T theta = C_2PI*((T)k) / (T)N;
        return std::polar2<T>(r, theta);
    };
    std::vector<complex_t<T>> twiddle_for_conj;
    for(size_t i=0;i<fft_len;i++){
        twiddle_for_conj.push_back(omega_func(i, fft_len));
    }

    if(correlation){
        size_t padding_size = fft_len-t_filter.size();
        for(size_t i=0;i<padding_size;i++){
            //t_filter.emplace_back((T)0, (T)0);
            t_filter.insert(t_filter.begin(),  complex_t<T>((T)0, (T)0));
        }
    }
#elif defined(USE_CORR_WARP_SHIFT)
    if(correlation){
        size_t padding_size = fft_len-t_filter.size();
        for(size_t i=0;i<padding_size;i++){
            //t_filter.emplace_back((T)0, (T)0);
            t_filter.insert(t_filter.begin(),  complex_t<T>((T)0, (T)0));
        }

        for(const auto & it : t_filter)
            std::cout<<it<<",";
        std::cout<<"\n";

        // wrap around
        // TODO: better solution
        // simple rotation to the right
        std::rotate(t_filter.rbegin(), t_filter.rbegin() + 1, t_filter.rend());
        for(const auto & it : t_filter)
            std::cout<<it<<",";
        std::cout<<"\n";
    }
#else
    if(correlation){
        std::reverse(t_filter.begin(), t_filter.end());
        for(size_t i=0;i<(fft_len-t_data.size());i++){
            t_filter.emplace_back((T)0, (T)0);
        }
    }
#endif
    else{
        for(size_t i=0;i<(fft_len-t_data.size());i++){
            t_filter.emplace_back((T)0, (T)0);
        }
    }


    fft_cooley_tukey(t_data, f_data, fft_len);
    fft_cooley_tukey(t_filter, f_filter, fft_len);

    if(correlation){
        for(size_t i=0;i<f_data.size();i++){
#if defined(USE_CORR_CONJ)
            f_data[i] = f_data[i] * (twiddle_for_conj[i] * std::conj(f_filter[i]));  // element-wise multiply
#elif defined(USE_CORR_WARP_SHIFT)
            f_data[i] = f_data[i] * f_filter[i];  // element-wise multiply
#else
            f_data[i] = f_data[i] * f_filter[i];  // element-wise multiply
#endif
        }
    }else{
        for(size_t i=0;i<f_data.size();i++){
            f_data[i] = f_data[i] * f_filter[i];  // element-wise multiply
        }
    }
    ifft_cooley_tukey(f_data, f_result);

    // crop to dst size
    for(size_t i=0;i<dst_len;i++)
        dst.push_back(std::real(f_result[i]));
}


void test_convolve_fft_1d(){
    const size_t filter_len = 3;
    const size_t data_len = 10;
    size_t i;

    std::vector<d_type> data;
    std::vector<d_type> filter;

    std::vector<d_type> output;
    std::vector<d_type> output_2;

    for(i=0;i<data_len;i++){
        data.push_back(i);
    }
    for(i=0;i<filter_len;i++){
        filter.push_back(i);
    }
    convolve_naive(data, filter, output, true);
    convolve_fft(data, filter, output_2, true);

    dump_vector(data);
    dump_vector(filter);
    dump_vector(output);
    dump_vector(output_2);

}


int main(){
    test_convolve_fft_1d();
#if 0
    size_t size;
    //size_t total_size = 1<<13;
    size_t total_size = 8;
    for(size = 2; size<=total_size; size *= 2){
        std::vector<complex_t<d_type>> t_seq;
        rand_vec(t_seq, size-1);

        std::vector<complex_t<d_type>> f_seq_1;
        std::vector<complex_t<d_type>> f_seq_2;
        std::vector<complex_t<d_type>> t_seq_1;
        std::vector<complex_t<d_type>> t_seq_2;
        fft_naive(t_seq, f_seq_1,size);
        ifft_naive(f_seq_1, t_seq_1,size);
        fft_cooley_tukey(t_seq, f_seq_2,size);
        ifft_cooley_tukey(f_seq_2, t_seq_2,size);

        int err_cnt = valid_vector(f_seq_1, f_seq_2);
        int ierr_cnt = valid_vector(t_seq_1, t_seq_2);
        std::cout<<"length:"<<size<<", fwd valid:"<< ( (err_cnt==0)?"y":"n" ) <<
            ", bwd valid:"<<( (ierr_cnt==0)?"y":"n" ) <<std::endl;
        dump_vector(t_seq);

        dump_vector(t_seq_1);
        dump_vector(t_seq_2);
        std::cout<<"---------------------------------------"<<std::endl;
    }
#endif
}