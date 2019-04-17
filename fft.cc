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
#include <string.h>
#include <functional>

typedef double d_type;

template<typename T>
void dump_vector(const std::vector<T> & vec){
    for(const T & elem : vec){
        std::cout<<elem<<", ";
    }
    std::cout<<std::endl;
}
template<typename T>
void dump_vector(const T * vec, size_t len){
    for(size_t i=0;i<len;i++){
        std::cout<<vec[i]<<", ";
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
int valid_vector(const std::vector<T> & lhs, const std::vector<T> & rhs, T delta = (T)0.001){
    assert(lhs.size() == rhs.size());
    size_t i;
    int err_cnt = 0;
    for(i = 0;i < lhs.size(); i++){
        T d = lhs[i]- rhs[i];
        d = ABS(d);
        if(d > delta){
            std::cout<<" diff at "<<i<<", lhs:"<<lhs[i]<<", rhs:"<<rhs[i]<<std::endl;
            err_cnt++;
        }
    }
    return err_cnt;
}

template<typename T>
void rand_vec(std::vector<complex_t<T>> & seq){
    static std::random_device rd;   // seed

    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(-2.0, 2.0);

    //seq.resize(len);
    size_t i;
    for(i=0;i<seq.size();i++){
        seq[i] = complex_t<T>(dist(mt), dist(mt));
    }
}
template<typename T>
void rand_vec(std::vector<T> & seq){
    static std::random_device rd;   // seed

    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(-2.0, 2.0);

    //seq.resize(len);
    size_t i;
    for(i=0;i<seq.size();i++){
        seq[i] =  dist(mt);
    }
}

template<typename T>
void copy_vec(std::vector<T> & src, std::vector<T> & dst){
    dst.resize(src.size());
    for(size_t i=0;i<src.size();i++){
        dst[i] = src[i];
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

int bit_reverse_nbits(int v, int nbits){
    int r = 0;
    int d = nbits-1;
    for(int i=0;i<nbits;i++){
        if(v & (1<<i))
            r |= 1<<d;
        d--;
    }
    return r;
}

template<typename ELEMENT_T>
void bit_reverse_radix2(ELEMENT_T * vec, size_t length){
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
void fft_cooley_tukey(complex_t<T> * seq, size_t length)
{
    if(length==1){
        //f_seq[0] = t_seq[0];
        return;
    }
    //http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    assert( ( (length & (length - 1)) == 0 ) && "current only length power of 2");

    auto omega_func = [](size_t total_n, size_t k){
        // e^( -1 * 2PI*k*n/N * i), here n is iter through each 
        T r = (T)1;
        T theta = -1 * C_2PI*k / total_n;
        return std::polar2<T>(r, theta);
    };
    bit_reverse_radix2(seq, length);

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
    size_t itr;
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
                auto t = omega_k * seq[k+itr/2];
                seq[k+itr/2] = seq[k] - t;
                seq[k] += t;
            }
        }
    }
}
template<typename T>
void ifft_cooley_tukey(complex_t<T> * seq, size_t length){
    if(length == 1)
        return;
    assert( ( (length & (length - 1)) == 0 ) && "current only length power of 2");
    bit_reverse_radix2(seq, length);

    auto omega_func_inverse = [](size_t total_n, size_t k){
        // e^(  2PI*k*n/N * i), here n is iter through each 
        T r = (T)1;
        T theta = C_2PI*k / total_n;
        return std::polar2<T>(r, theta);
    };
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
    size_t itr;
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
                auto t = omega_k * seq[k+itr/2];
                seq[k+itr/2] = seq[k] - t;
                seq[k] += t;
            }
        }
    }

    // inverse only, need devide
    for(itr = 0; itr < length; itr++)
        seq[itr] /= (T)length;
}

template<typename T>
void _fft_cooley_tukey_r(complex_t<T> * seq, size_t length, bool is_inverse_fft){
    if(length == 1)
        return;
    assert( ( (length & (length - 1)) == 0 ) && "current only length power of 2");

    std::function<complex_t<T>(size_t,size_t)> omega_func;
    if(is_inverse_fft){
        omega_func = [](size_t total_n, size_t k){
            T r = (T)1;
            T theta = C_2PI*k / total_n;
            return std::polar2<T>(r, theta);
        };
    }else{
        omega_func = [](size_t total_n, size_t k){
            T r = (T)1;
            T theta = -1*C_2PI*k / total_n;
            return std::polar2<T>(r, theta);
        };
    }

    /*
    * length
    *   2    ->  0, 1
    *   4    ->  0, 2, 1, 3
    *   8    ->  0, 4, 2, 6, 1, 5, 3, 7
    *  16    ->  0, 8, 4,12, 2,10, 6,14, 1, 9, 5,13, 3,11, 7,15
    */
   for(size_t itr = 2; itr<=length; itr<<=1){
        size_t stride = length/itr;
        size_t groups = itr/2;
        size_t group_len = stride*2;

        std::vector<complex_t<T>> omega_list;   // pre-compute omega, and index to it later
        omega_list.resize(itr/2);
        for(size_t i = 0; i < itr/2 ; i ++){
            omega_list[i] = omega_func( itr, i);
        }
        for(size_t g=0;g<groups;g++){
            size_t k = bit_reverse_nbits(g, std::log2(groups));

            complex_t<T> & omega = omega_list[k];
            for(size_t s=0;s<stride;s++){
                //printf("W%d_%d(%d,%d-%d) ", itr, k, g, g*group_len+s, g*group_len+s+stride);
                complex_t<T> & a = seq[g*group_len+s];
                complex_t<T> & b = seq[g*group_len+s+stride];

#define BUFL2(a,b,w) \
    do{             \
        complex_t<T> temp = a;  \
        a = a + b*w;            \
        b = temp-b*w;           \
    }while(0)

                BUFL2(a, b, omega);

            }
        }
        //printf("\n");
    }

    // no forget last bit reverse!!
    bit_reverse_radix2(seq, length);
    if(is_inverse_fft){
        for(size_t i=0;i<length;i++)
            seq[i] = seq[i]/length;
    }
}
template<typename T>
void fft_cooley_tukey_r(complex_t<T> * seq, size_t length){
    _fft_cooley_tukey_r(seq, length,false);
}
template<typename T>
void ifft_cooley_tukey_r(complex_t<T> * seq, size_t length){
    _fft_cooley_tukey_r(seq, length,true);
}
/*
* http://processors.wiki.ti.com/index.php/Efficient_FFT_Computation_of_Real_Input
*
* r2c:
* N=length
* 1. input real g(n), len:N, form N/2 complex sequency x(n), len:N/2
*    xr(n) = g(2*n)
*    xi(n) = g(2*n+1)
* 2. compute N/2 point fft, x(n)->X(k), len:N/2
* 3. get final G(k) len:N, from X(k) len:N/2
*    a) for first half:
*      G(k) = X(k)A(k)+X*(N-k)B(k), k:0...N/2-1
*                   and, let X(N) = X(0)
*           A(k) = 0.5*(1-j*W(N,k)), k:0...N/2-1
*           B(k) = 0.5*(1+j*W(N,k)), k:0...N/2-1
*           W(N,k) = e^( -1 * 2PI*k/N * j)
*    b) for second half:
*      Gr(N/2) = Xr(0) - Xi(0),    real - imag
*      Gi(N/2) = 0
*      G(N-k) = G*(k), k:1...N/2-1
*
*
* step 3 can re-write as follow:
*   Ar(k) = 0.5*(1.0-sin(2*PI*k/N))
*   Ai(k) = 0.5*(-1*cos(2*PI*k/N))
*   Br(k) = 0.5*(1+sin(2*PI*k/N))
*   Bi(k) = 0.5*(1*cos(2*PI*k/N))
*               k=0...N/2-1
*
*   a) for first half:
*   Gr(k) = Xr(k)Ar(k) – Xi(k)Ai(k) + Xr(N/2–k)Br(k) + Xi(N/2–k)Bi(k)
*   Gi(k) = Xi(k)Ar(k) + Xr(k)Ai(k) + Xr(N/2–k)Bi(k) – Xi(N/2–k)Br(k)
*                   for k = 0...N/2–1 and X(N/2) = X(0)
*
*   Gr(N/2) = Xr(0) – Xi(0)
*   Gi(N/2) = 0
*   Gr(N–k) = Gr(k), for k = 1...N/2–1
*   Gi(N–k) = –Gi(k)
*/
template<typename T>
void fft_r2c(T* t_seq, complex_t<T> * f_seq, size_t length){
    if(length == 1)
        return;
    assert( ( (length & (length - 1)) == 0 ) && "current only length power of 2");

    auto omega_func = [](size_t total_n, size_t k){
        // e^( -1 * 2PI*k*n/N * i), here n is iter through each 
        T r = (T)1;
        T theta = -1 * C_2PI*k / total_n;
        return std::polar2<T>(r, theta);
    };

    std::vector<complex_t<T>> A;
    std::vector<complex_t<T>> B;
    for(size_t i=0;i<length/2;i++){
        complex_t<T> v (0,1);
        complex_t<T> r (1,0);
        v*=omega_func(length,i);
        A.push_back( (r-v)*0.5 );
        B.push_back( (r+v)*0.5 );
    }

    std::vector<complex_t<T>> seq;
    for(size_t i=0;i<length/2;i++){
        seq.emplace_back(t_seq[2*i], t_seq[2*i+1]);
    }
    fft_cooley_tukey_r(seq.data(), length/2);

    f_seq[0] = seq[0]*A[0]+std::conj(seq[0])*B[0];    // X(N/2)=X(0)
    for(size_t i=1;i<length/2;i++){
        f_seq[i] = seq[i] *A[i]+std::conj(seq[length/2-i])*B[i];
        f_seq[length-i] = std::conj(f_seq[i]);
    }
    f_seq[length/2] = complex_t<T>( std::real(seq[0])-std::imag(seq[0]), (T)0);
}

/*
* http://processors.wiki.ti.com/index.php/Efficient_FFT_Computation_of_Real_Input
*
* c2r:
* N=length
* 1. input G(k), len:N, form N/2 complex sequency X(k), len:N/2
*    X(k) = G(k)A*(k) + G*(N/2-k)B*(k), k:0...N/2-1
*           A(k) = 0.5*(1-j*W(N,k)), k:0...N/2-1
*           B(k) = 0.5*(1+j*W(N,k)), k:0...N/2-1
*           W(N,k) = e^( -1 * 2PI*k/N * j)
*           A(k), B(k), same as r2c
* 2. compute N/2 point ifft, X(k)->x(n), len:N/2
* 3. get final real g(n) len:N, from x(n) len:N/2
*      g(2*n)   = xr(n)
*      g(2*n+1) = xi(n)
*               n=0...N/2-1
*
* step 1 can re-write:
*   Xr(k) = Gr(k)IAr(k) – Gi(k)IAi(k) + Gr(N/2–k)IBr(k) + Gi(N/2–k)IBi(k)
*   Xi(k) = Gi(k)IAr(k) + Gr(k)IAi(k) + Gr(N/2–k)IBi(k) – Gi(N/2–k)IBr(k)
*                for k = 0...N/2–1
*
*   IA : complex conjugate of A
*   IB : complex conjugate of B
*   IAr(k) = 0.5*(1.0-sin(2*PI*k/N))
*   IAi(k) = 0.5*(1*cos(2*PI*k/N))
*   IBr(k) = 0.5*(1+sin(2*PI*k/N))
*   IBi(k) = 0.5*(-1*cos(2*PI*k/N))
*               k=0...N/2-1
*/
template<typename T>
void fft_c2r(complex_t<T> * f_seq, T* t_seq, size_t length){
    if(length == 1)
        return;
    assert( ( (length & (length - 1)) == 0 ) && "current only length power of 2");

    auto omega_func = [](size_t total_n, size_t k){
        // e^( -1 * 2PI*k*n/N * i), here n is iter through each 
        T r = (T)1;
        T theta = -1 * C_2PI*k / total_n;
        return std::polar2<T>(r, theta);
    };

    std::vector<complex_t<T>> A;
    std::vector<complex_t<T>> B;
    for(size_t i=0;i<length/2;i++){
        complex_t<T> v (0,1);
        complex_t<T> r (1,0);
        v*=omega_func(length,i);
        A.push_back( (r-v)*0.5 );
        B.push_back( (r+v)*0.5 );
    }

    std::vector<complex_t<T>> seq;
    seq.resize(length/2);

    for(size_t itr = 0; itr<length/2; itr++){
        seq[itr] = f_seq[itr]*std::conj(A[itr])+std::conj(f_seq[length/2-itr])*std::conj(B[itr]);
    }

    ifft_cooley_tukey_r(seq.data(), length/2);

    for(size_t i=0;i<length/2;i++){
        t_seq[2*i] = std::real(seq[i]);
        t_seq[2*i+1] = std::imag(seq[i]);
    }
}

template<typename T>
void fft_2d(complex_t<T>* seq, size_t seq_w, size_t seq_h)
{
    size_t i,j;
    for(i=0;i<seq_h;i++){
        fft_cooley_tukey(seq+i*seq_w, seq_w);
    }

    std::vector<complex_t<T>> s2;
    // transpose
    for(i=0;i<seq_w;i++){
        for(j=0;j<seq_h;j++){
            s2.push_back(seq[j*seq_w+i]);
        }
    }
    for(i=0;i<seq_w;i++){
        fft_cooley_tukey(s2.data()+i*seq_h, seq_h);
    }
    for(i=0;i<seq_w;i++){
        for(j=0;j<seq_h;j++){
            seq[j*seq_w+i] = s2[i*seq_h+j];
        }
    }
}
template<typename T>
void ifft_2d(complex_t<T>* seq, size_t seq_w, size_t seq_h)
{
    size_t i,j;
    for(i=0;i<seq_h;i++){
        ifft_cooley_tukey(seq+i*seq_w, seq_w);
    }

    std::vector<complex_t<T>> s2;
    // transpose
    for(i=0;i<seq_w;i++){
        for(j=0;j<seq_h;j++){
            s2.push_back(seq[j*seq_w+i]);
        }
    }
    for(i=0;i<seq_w;i++){
        ifft_cooley_tukey(s2.data()+i*seq_h, seq_h);
    }
    for(i=0;i<seq_w;i++){
        for(j=0;j<seq_h;j++){
            seq[j*seq_w+i] = s2[i*seq_h+j];
        }
    }
}

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
    for(size_t p=0;p<pad;p++){
        d.insert(d.begin(), (T)0);
        d.push_back((T)0);
    }

    dst.reserve(dst_len);

    for(i=0;i<dst_len;i++){
        T v = 0;
        for(j=0;j<filter.size();j++){
            v += f[j] * d[i+j];
        }
        dst.push_back(v);
    }
}

template<typename T>
void convolve2d_naive(const T* data, size_t data_w, size_t data_h,
    const T* filter, size_t filter_w, size_t filter_h,
    T* dst, bool correlation = false)
{
    size_t dst_h = data_h + filter_h - 1;
    size_t dst_w = data_w + filter_w - 1;
    size_t pad_h = filter_h -1;
    size_t pad_w = filter_w -1;
    size_t i,j,ki,kj;

    std::vector<T> _ff;
    const T * f = filter;

    if(!correlation){
        _ff.resize(filter_w*filter_h);
        std::reverse_copy(filter, filter+filter_w*filter_h,_ff.begin());
        f = _ff.data();
    }
    //memset(dst, 0, dst_w*dst_h*sizeof(T));

    auto get_data_value=[&](size_t dh, size_t dw){
        size_t h, w;
        h = dh-pad_h;
        w = dw-pad_w;
        if(dh < pad_h || h >= data_h)
            return (T)0;
        if(dw < pad_w || w >= data_w)
            return (T)0;
        size_t idx = h * data_w + w;
        return data[idx];
    };
    for(j=0;j<dst_h;j++){
        for(i=0;i<dst_w;i++){
            T v = 0;
            for(kj=0;kj<filter_h;kj++){
                for(ki=0;ki<filter_w;ki++){
                    v += f[kj*filter_w+ki] * get_data_value(j+kj, i+ki);
                }
            }
            dst[j*dst_w+i] = v;
        }
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
*
*    in fact, if use time reverse plus shift in DFT, the twiddle factor is not needed.
*    [0,1,2,3,4] --- reverse ---> [4,3,2,1,0] --- shr ---> [0,4,3,2,1]
*/

// convolve_xx() function vehavior same as numpy:
/*
* data=np.array(...)    # some 1d array
* filter = np.array(...) # some 1d array
* len_data=data.shape[0]
* len_filter=filter.shape[0]
* len_out = len_data+len_filter-1
* dp = np.pad(data, (0,len_out-len_data),'constant')
* fp = np.pad(filter,(0,len_out-len_filter),'constant')
* fft_out = np.fft.ifft(np.fft.fft(dp)*np.fft.fft(fp))
*/
// np.correlate(data, filter, 'full')

//#define USE_CORR_CONJ
//#define USE_CORR_WARP_SHIFT
template<typename T>
void convolve_fft(const std::vector<T> & data, const std::vector<T> & filter, std::vector<T> &  dst, bool correlation = false){
    size_t dst_len = data.size()+filter.size()-1;

    // round to nearest 2^power number
    size_t fft_len = (size_t)std::pow(2, std::ceil(std::log2(dst_len)));
    std::vector<complex_t<T>> seq_data;
    std::vector<complex_t<T>> seq_filter;

    seq_data.reserve(fft_len);
    seq_filter.reserve(fft_len);

    for(const auto & it :data)
        seq_data.emplace_back(it , (T)0);
    for(const auto & it :filter)
        seq_filter.emplace_back(it , (T)0);

    // zero padding
    complex_t<T> c_zero((T)0, (T)0);
    seq_data.resize(fft_len, c_zero);

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
        size_t padding_size = fft_len-seq_filter.size();
        for(size_t i=0;i<padding_size;i++){
            //seq_filter.emplace_back((T)0, (T)0);
            seq_filter.insert(seq_filter.begin(),  complex_t<T>((T)0, (T)0));
        }
    }
#elif defined(USE_CORR_WARP_SHIFT)
    if(correlation){
        size_t padding_size = fft_len-seq_filter.size();
        for(size_t i=0;i<padding_size;i++){
            seq_filter.insert(seq_filter.begin(),  complex_t<T>((T)0, (T)0));
        }

        // wrap around
        // TODO: better solution
        // simple rotation to the right
        std::rotate(seq_filter.rbegin(), seq_filter.rbegin() + 1, seq_filter.rend());
    }
#else
    if(correlation){
        std::reverse(seq_filter.begin(), seq_filter.end());
        for(size_t i=0;i<(fft_len-seq_filter.size());i++){
            //seq_filter.insert(seq_filter.begin(), complex_t<T>((T)0, (T)0));
            seq_filter.emplace_back((T)0, (T)0);
        }
    }
#endif
    else{
        for(size_t i=0;i<(fft_len-seq_filter.size());i++){
            seq_filter.emplace_back((T)0, (T)0);
        }
    }

    fft_cooley_tukey(seq_data.data(), fft_len);
    fft_cooley_tukey(seq_filter.data(), fft_len);

    if(correlation){
        for(size_t i=0;i<seq_data.size();i++){
#if defined(USE_CORR_CONJ)
            seq_data[i] = seq_data[i] * (twiddle_for_conj[i] * std::conj(seq_filter[i]));  // element-wise multiply
#elif defined(USE_CORR_WARP_SHIFT)
            seq_data[i] = seq_data[i] * std::conj(seq_filter[i]);  // element-wise multiply
#else
            seq_data[i] = seq_data[i] * seq_filter[i];  // element-wise multiply
#endif
        }
    }else{
        for(size_t i=0;i<seq_data.size();i++){
            seq_data[i] = seq_data[i] * seq_filter[i];  // element-wise multiply
        }
    }
    ifft_cooley_tukey(seq_data.data(), fft_len);

    // crop to dst size
    for(size_t i=0;i<dst_len;i++)
        dst.push_back(std::real(seq_data[i]));
}

template<typename T>
void convolve2d_fft(const T* data, size_t data_w, size_t data_h,
    const T* filter, size_t filter_w, size_t filter_h,
    T* dst, bool correlation = false)
{
    size_t dst_w = data_w + filter_w - 1;
    size_t dst_h = data_h + filter_h - 1;

    // round to nearest 2^power number
    size_t fft_len_w = (size_t)std::pow(2, std::ceil(std::log2(dst_w)));
    size_t fft_len_h = (size_t)std::pow(2, std::ceil(std::log2(dst_h)));
    std::vector<complex_t<T>> seq_data;
    std::vector<complex_t<T>> seq_filter;

    complex_t<T> c_zero((T)0, (T)0);
    seq_data.resize(fft_len_w*fft_len_h, c_zero );
    seq_filter.resize(fft_len_w*fft_len_h, c_zero );

    // padding filter
#if defined( USE_CORR_CONJ )
#elif defined(USE_CORR_WARP_SHIFT)
    if(correlation){
        size_t pad_w = fft_len_w-filter_w;
        size_t pad_h = fft_len_h-filter_h;
        seq_filter[0] = complex_t<T>(filter[(filter_h-1)*filter_w+filter_w-1], T(0));
        for(size_t i=0;i<filter_w-1;i++){
            seq_filter[pad_w+i+1] = complex_t<T>(filter[(filter_h-1)*filter_w+i], T(0));
        }
        for(size_t j=0;j<fft_len_h-1;j++){
            if(j<filter_h-1){
                seq_filter[(j+1)*fft_len_w] = complex_t<T>(filter[j*filter_w+filter_w-1], T(0));
                for(size_t i=0;i<filter_w-1;i++){
                    seq_filter[(j+1)*fft_len_w+pad_w+i+1] = complex_t<T>(filter[j*filter_w+i], T(0));
                }
            }
        }
    }
#else
    if(correlation){
        size_t pad_w = fft_len_w-filter_w;
        size_t pad_h = fft_len_h-filter_h;
        std::vector<complex_t<T>> tmp_filter;
        for(size_t j=0;j<filter_w*filter_h;j++){
            tmp_filter.emplace_back(filter[j], T(0));
        }
        std::reverse(tmp_filter.begin(), tmp_filter.end());
        for(size_t j=0;j<fft_len_h;j++){
            if(j<filter_h){
                for(size_t i=0;i<filter_w;i++){
                    seq_filter[j*fft_len_w+i] = tmp_filter[j*filter_w+i];
                }
            }
        }
    }
#endif
    else{
        for(size_t j=0;j<fft_len_h;j++){
            if(j<filter_h){
                for(size_t i=0;i<filter_w;i++){
                    seq_filter[j*fft_len_h+i] = complex_t<T>(filter[j*filter_w+i], T(0));
                }
            }
        }
    }

    // padding data
    for(size_t j=0;j<fft_len_h;j++){
        if(j<data_h){
            for(size_t i=0;i<data_w;i++){
                seq_data[j*fft_len_h+i] = complex_t<T>(data[j*data_w+i], T(0));
            }
        }
    }

    fft_2d(seq_data.data(), fft_len_w, fft_len_h);
    fft_2d(seq_filter.data(), fft_len_w, fft_len_h);

    // element-wise multiply
    if(correlation){
        for(size_t i=0;i<seq_data.size();i++){
#if defined( USE_CORR_CONJ )
#elif defined(USE_CORR_WARP_SHIFT)
            seq_data[i] = seq_data[i] * std::conj(seq_filter[i]);
#else
            seq_data[i] = seq_data[i] * seq_filter[i];
#endif
        }
    }
    else{
        for(size_t i=0;i<seq_data.size();i++){
            seq_data[i] = seq_data[i] * seq_filter[i];
        }
    }

    ifft_2d(seq_data.data(), fft_len_w, fft_len_h);
    // crop to dst size
    for(size_t j=0;j<dst_h;j++){
        for(size_t i=0;i<dst_w;i++){
            dst[j*dst_w + i] = std::real(seq_data[j*fft_len_w+i]);
        }
    }
}

void test_convolve_fft_1d(){
    const size_t filter_len = 9;
    const size_t data_len = 16;

    std::vector<d_type> data;
    std::vector<d_type> filter;

    auto test_convolve_func = [](
        const std::vector<d_type> & data, 
        const std::vector<d_type> & filter,
        const bool is_correlate)
    {
        std::vector<d_type> output;
        std::vector<d_type> output_2;
        convolve_naive(data, filter, output, is_correlate);
        convolve_fft(data, filter, output_2, is_correlate);
        int err_cnt = valid_vector(output, output_2);
        std::cout<<  (is_correlate? "[corr]":"[conv]")<<
            "data size:"<<data.size()<<", filter size:"<<filter.size()<<
            ", result valid:"<<( (err_cnt==0)?"y":"n" )<<std::endl;
    };
#if 0
    for(size_t i=0;i<data_len;i++){
        data.push_back(i);
    }
    for(size_t i=0;i<filter_len;i++){
        filter.push_back(i);
    }
#endif
    data.resize(data_len);
    filter.resize(filter_len);
    rand_vec(data);
    rand_vec(filter);

    test_convolve_func(data, filter, true);
    test_convolve_func(data, filter, false);
}

void test_convolve_fft_2d(){
    const size_t filter_h = 4;
    const size_t filter_w = 4;
    const size_t data_h = 5;
    const size_t data_w = 5;

    std::vector<d_type> data;
    std::vector<d_type> filter;

    data.reserve(data_h*data_w);
    filter.reserve(filter_h*filter_w);

    for(size_t i=0;i<(data_h*data_w);i++){
        data.push_back(i);
    }
    for(size_t i=0;i<(filter_h*filter_w);i++){
        filter.push_back(i);
    }

    auto test_convolve2d_func = [](
        const std::vector<d_type> & data, size_t data_w, size_t data_h,
        const std::vector<d_type> & filter, size_t filter_w, size_t filter_h,
        const bool is_correlate)
    {
        std::vector<d_type> output;
        std::vector<d_type> output_2;
        size_t out_w = data_w + filter_w - 1;
        size_t out_h = data_h + filter_h - 1;
        output.resize(out_w*out_h);
        output_2.resize(out_w*out_h);

        convolve2d_naive(data.data(), data_w, data_h,
                    filter.data(), filter_w, filter_h,
                    output.data(), is_correlate);
        convolve2d_fft(data.data(), data_w, data_h,
                    filter.data(), filter_w, filter_h,
                    output_2.data(), is_correlate);
        int err_cnt = valid_vector(output, output_2);
        std::cout<<  (is_correlate? "[corr2d]":"[conv2d]")<<
            "data size:"<<data.size()<<", filter size:"<<filter.size()<<
            ", result valid:"<<( (err_cnt==0)?"y":"n" )<<std::endl;
        //dump_vector(data);
        //dump_vector(filter);
        //dump_vector(output);
        //dump_vector(output_2);
    };

    test_convolve2d_func(data, data_w, data_h, filter, filter_w, filter_h, true );
    test_convolve2d_func(data, data_w, data_h, filter, filter_w, filter_h, false);
#if 0
    {
        std::vector<complex_t<d_type>> v;
        for(auto & it : filter)
            v.emplace_back(  it, (d_type)0);
        fft_2d(v.data(), filter_w, filter_h);
        dump_vector(v);
        ifft_2d(v.data(), filter_w, filter_h);
        dump_vector(v);
    }
#endif
}

int main(){
    test_convolve_fft_1d();
    test_convolve_fft_2d();
#if 1
    size_t size;
    //size_t total_size = 1<<13;
    size_t total_size = 1024;
    for(size = 2; size<=total_size; size *= 2){
        std::vector<complex_t<d_type>> t_seq;
        std::vector<complex_t<d_type>> f_seq;
        std::vector<complex_t<d_type>> t_seq_r;

        std::vector<complex_t<d_type>> seq_fwd;
        std::vector<complex_t<d_type>> seq_bwd;
        t_seq.resize(size);
        rand_vec(t_seq);
        copy_vec(t_seq,seq_fwd);

        fft_naive(t_seq, f_seq, size);
        ifft_naive(f_seq, t_seq_r,size);

        //fft_cooley_tukey(seq_fwd.data() ,size);
        fft_cooley_tukey_r(seq_fwd.data() ,size);
        int err_cnt = valid_vector(f_seq, seq_fwd);

        copy_vec(f_seq,seq_bwd);
        ifft_cooley_tukey_r(seq_bwd.data(), size);
        int ierr_cnt = valid_vector(t_seq_r, seq_bwd);
        std::cout<<"length:"<<size<<", fwd valid:"<< ( (err_cnt==0)?"y":"n" ) <<
            ", bwd valid:"<<( (ierr_cnt==0)?"y":"n" ) <<std::endl;
        //dump_vector(t_seq);
        //dump_vector(t_seq_r);
        std::cout<<"---------------------------------------"<<std::endl;
    }
    for(size = 2; size<=total_size; size *= 2){
        std::vector<complex_t<d_type>> t_seq;
        std::vector<complex_t<d_type>> f_seq;
        std::vector<complex_t<d_type>> t_seq_r;

        std::vector<d_type> seq_fwd_real;
        std::vector<complex_t<d_type>> seq_fwd;
        std::vector<complex_t<d_type>> seq_bwd;
        std::vector<d_type> seq_bwd_real;
        seq_fwd_real.resize(size);
        rand_vec(seq_fwd_real);
        for(size_t ii=0;ii<size;ii++){
            t_seq.push_back(complex_t<d_type>(seq_fwd_real[ii],(d_type)0));
        }

        fft_naive(t_seq, f_seq, size);
        ifft_naive(f_seq, t_seq_r,size);

        //fft_cooley_tukey(seq_fwd.data() ,size);
        seq_fwd.resize(size);
        fft_r2c(seq_fwd_real.data(), seq_fwd.data() ,size);
        int err_cnt = valid_vector(f_seq, seq_fwd);

        //copy_vec(f_seq,seq_bwd);
        //ifft_cooley_tukey(seq_bwd.data(), size);
        seq_bwd_real.resize(size);
        fft_c2r(f_seq.data(), seq_bwd_real.data(), size);
        for(size_t ii=0;ii<size;ii++){
            seq_bwd.push_back(complex_t<d_type>(seq_bwd_real[ii],(d_type)0));
        }
        int ierr_cnt = valid_vector(t_seq_r, seq_bwd);
        std::cout<<"length:"<<size<<", r2c fwd valid:"<< ( (err_cnt==0)?"y":"n" ) <<
            ", c2r bwd valid:"<<( (ierr_cnt==0)?"y":"n" ) <<std::endl;
        //dump_vector(t_seq);
        //dump_vector(f_seq);
        //dump_vector(seq_fwd);
        //dump_vector(t_seq_r);
        std::cout<<"---------------------------------------"<<std::endl;
    }
#endif
}