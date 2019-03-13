#include "complex.h"
#include <vector>
#include <complex>
#include <stddef.h>


// https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html
template<typename T>
void fft_naive(const std::vector<T> & in, std::vector<complex_t<T>> & out, size_t length=0){
    // https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Definition
    auto omega_func = [](size_t total_n, size_t k){
        // e^( -1 * 2PI*k*n/N * i), here n is iter through each 
        T r = (T)1;
        T theta = -1 * 2*C_PI*k / total_n;
        return std::polar2<T>(r, theta);
    };
    if(length == 0)
        length = in.size();
    size_t fft_n = length;
    size_t k;
    std::vector<T> seq = in;
    if(length > in.size()){
        for(size_t i=0; i< (length - in.size()); i++ ){
            seq.push_back((T)0);
        }
    }
    for(k=0;k<fft_n;k++){
        size_t n;
        complex_t<T> omega_k = omega_func(fft_n, k);
        complex_t<T> A_k;
        for(n=0;n<fft_n;n++){
            A_k +=  std::pow(omega_k, (T)n) * in[n] ;
        }
        out.push_back(A_k);
    }
}

int main(){
    size_t size = 8;
    std::vector<double> time_dom;
    for(size_t i=0;i<size;i++){
        time_dom.push_back(i);
    }
    std::vector<complex_t<double>> freq_dom;
    fft_naive(time_dom, freq_dom, 6);
    for(size_t i=0;i<size;i++){
        std::cout<<time_dom[i]<<" ";
    }
    std::cout<<std::endl;

    for(size_t i=0;i<freq_dom.size();i++){
        std::cout<<freq_dom[i]<<" ";
    }
    std::cout<<std::endl;
}