#include "complex.h"
#include <vector>
#include <complex>
#include <stddef.h>
#include <utility> // std::swap in c++11
#include <assert.h>
#include <iostream>

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

// https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html
template<typename T>
void fft_naive(const std::vector<T> & t_seq, std::vector<complex_t<T>> & f_seq, size_t length=0){
    // https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Definition
    auto omega_func = [](size_t total_n, size_t k){
        // e^( -1 * 2PI*k*n/N * i), here n is iter through each 
        T r = (T)1;
        T theta = -1 * 2*C_PI*k / total_n;
        return std::polar2<T>(r, theta);
    };
    if(length == 0)
        length = t_seq.size();
    size_t fft_n = length;
    size_t k;
    std::vector<T> seq = t_seq;
    if(length > t_seq.size()){
        for(size_t i=0; i< (length - t_seq.size()); i++ ){
            seq.push_back((T)0);
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
        T theta = 2*C_PI*k / total_n;
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
    std::vector<size_t> r_idx;
    bit_reverse_permute(std::log2(length), r_idx);
    size_t i;
    size_t ir;
    for(i=0;i<length;i++){
        ir = r_idx[i];
        //std::cout<<"i:"<<i<<", ir:"<<ir<<std::endl;
        if(i<ir)
            std::swap(vec[i], vec[ir]);
    }
}

template<typename T>
void fft_cooley_tukey(const std::vector<T> & t_seq, std::vector<complex_t<T>> & f_seq, size_t length=0)
{
    auto omega_func = [](size_t total_n, size_t k){
        // e^( -1 * 2PI*k*n/N * i), here n is iter through each 
        T r = (T)1;
        T theta = -1 * 2*C_PI*k / total_n;
        return std::polar2<T>(r, theta);
    };

    // TODO, length or t_seq.size() is radix of 2
    if(length == 0)
        length = t_seq.size();
    size_t itr;

    if(length == 1){
        f_seq.emplace_back(t_seq[0], (T)0);
        return ;
    }

    //http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    assert( ( (length & (length - 1)) == 0 ) && "current only length power of 2");

    for(const T & s : t_seq){
        f_seq.emplace_back(s, (T)0);
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
        T theta = 2*C_PI*k / total_n;
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

int main(){
    size_t size;
    size_t total_size = 1<<13;
    //size_t total_size = 8;
    for(size = 2; size<=total_size; size *= 2){
        std::vector<d_type> t_seq;
        for(size_t i=0;i<size;i++){
            t_seq.push_back(i);
        }
        std::vector<complex_t<d_type>> f_seq_1;
        std::vector<complex_t<d_type>> f_seq_2;
        std::vector<complex_t<d_type>> t_seq_1;
        std::vector<complex_t<d_type>> t_seq_2;
        fft_naive(t_seq, f_seq_1);
        ifft_naive(f_seq_1, t_seq_1);
        fft_cooley_tukey(t_seq, f_seq_2);
        ifft_cooley_tukey(f_seq_2, t_seq_2);

        int err_cnt = valid_vector(f_seq_1, f_seq_2);
        int ierr_cnt = valid_vector(t_seq_1, t_seq_2);
        std::cout<<"length:"<<size<<", fwd valid:"<< ( (err_cnt==0)?"y":"n" ) <<
            ", bwd valid:"<<( (ierr_cnt==0)?"y":"n" ) <<std::endl;
        //dump_vector(t_seq);

        //dump_vector(t_seq_1);
        //dump_vector(t_seq_2);
        //std::cout<<"---------------------------------------"<<std::endl;
    }
}