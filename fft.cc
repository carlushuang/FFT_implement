#include "complex.h"
#include <vector>
#include <complex>
#include <stddef.h>


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

typedef double d_type;

int main(){
    size_t size = 8;
    std::vector<d_type> time_dom;
    for(size_t i=0;i<size;i++){
        time_dom.push_back(i);
    }
    std::vector<complex_t<d_type>> freq_dom;
    fft_naive(time_dom, freq_dom, 6);
    for(size_t i=0;i<size;i++){
        std::cout<<time_dom[i]<<", ";
    }
    std::cout<<std::endl;

    for(size_t i=0;i<freq_dom.size();i++){
        std::cout<<freq_dom[i]<<", ";
    }
    std::cout<<std::endl;

    std::vector<complex_t<d_type>> t_dom_i;
    ifft_naive(freq_dom,  t_dom_i, 6);
    for(size_t i=0;i<t_dom_i.size();i++){
        std::cout<<t_dom_i[i]<<", ";
    }
    std::cout<<std::endl;
}