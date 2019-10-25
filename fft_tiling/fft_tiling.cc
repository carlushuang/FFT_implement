#include <stdio.h>
#include <random>
#include <stdlib.h>
#include <limits>
#include <iostream>

#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <functional>
#include <tuple>

#define LD_C(vec,idx,r,i) do{r=vec[2*(idx)];i=vec[2*(idx)+1];}while(0)
#define ST_C(vec,idx,r,i) do{vec[2*(idx)]=r;vec[2*(idx)+1]=i;}while(0)
#define BTFL_C(ar,ai,br,bi,omr,omi,tr,ti) do{\
    tr=br*omr-bi*omi; ti=br*omi+bi*omr; br=ar; bi=ai;\
    ar=ar+tr;ai=ai+ti; br=br-tr;bi=bi-ti; } while(0)
#define IBTFL_C(ar,ai,br,bi,omr,omi,tr,ti) do{\
    tr=br*omr+bi*omi; ti=-br*omi+bi*omr; br=ar; bi=ai;\
    ar=ar+tr;ai=ai+ti; br=br-tr;bi=bi-ti; } while(0)
#define R2C_EPILOG(gr,gi,gnr,gni,s,c,tr0,ti0,tr1,ti1) do{ \
    tr0=gr+gnr; ti0=gr-gnr; tr1=gi+gni; ti1=gi-gni;\
    gr = 0.5*(tr0 - ti0*s + tr1*c); gi = 0.5*(ti1 - tr1*s - ti0*c); \
    gnr = 0.5*(tr0 + ti0*s - tr1*c); gni = -0.5*(ti1 + tr1*s + ti0*c); }while(0)
#define IC2R_EPILOG(xr,xi,xnr,xni,s,c,sr0,si0,sr1,si1)  do{             \
    sr0=xr+xnr; si0=xr-xnr; sr1=xi+xni; si1=xi-xni;                     \
    xr = 0.5*(sr0 + si0*s - sr1*c);  xi = 0.5*(si1 + sr1*s + si0*c);    \
    xnr = 0.5*(sr0 - si0*s + sr1*c); xni = 0.5*(-1*si1 + sr1*s + si0*c); }while(0)

#ifndef C_PI
#define C_PI  3.14159265358979323846
#endif
#ifndef C_2PI
#define C_2PI 6.28318530717958647692
#endif
static inline int bit_reverse_nbits(int v, int nbits){
    int r = 0; int d = nbits-1;
    for(int i=0;i<nbits;i++) { if(v & (1<<i)) r |= 1<<d;  d--; }
    return r;
}
// below function produce  https://oeis.org/A030109
static inline void bit_reverse_permute(size_t radix2_num, std::vector<size_t> &arr)
{
    arr.resize(pow(2,radix2_num)); arr[0] = 0;
    for(size_t k=0;k<radix2_num;k++){ size_t last_k_len = pow(2, k);  size_t last_k;
       for(last_k = 0; last_k < last_k_len; last_k++){
           arr[last_k] = 2*arr[last_k]; arr[last_k_len+last_k] = arr[last_k]+1;}  }
}
template<typename T>
static inline void bit_reverse_radix2_c(T *vec,size_t c_length){
    assert( ( (c_length & (c_length - 1)) == 0 ) && "must be radix of 2");
    std::vector<size_t> r_idx;
    bit_reverse_permute(log2(c_length), r_idx);
    for(size_t i=0;i<c_length;i++){  size_t ir = r_idx[i];
        if(i<ir) { std::swap(vec[2*i], vec[2*ir]); std::swap(vec[2*i+1], vec[2*ir+1]); } }
}
static inline int64_t fft_conv_out_size(int64_t in_size, int64_t pad, int64_t dilation, int64_t ksize, int64_t stride)
{
     return (in_size + 2*pad- dilation*(ksize-1) -1)/stride + 1;
}
template<typename T>
static inline void _fft_cooley_tukey_r_mt_shifted(T * seq, size_t c_length, size_t phase_n, size_t phase_k, bool is_inverse_fft, bool need_final_reverse=true){
    if(c_length == 1) return;
    assert( ( (c_length & (c_length - 1)) == 0 ) && "current only length power of 2");

    std::function<std::tuple<T,T>(size_t,size_t)> omega_func = [](size_t total_n, size_t k){
            T theta = -1*C_2PI*k / total_n;
            return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta)); };

    for(size_t itr = 2; itr<=c_length; itr<<=1){
        size_t stride = c_length/itr;
        size_t groups = itr/2;
        size_t group_len = stride*2;
        std::tuple<T,T> phase = omega_func(phase_n,phase_k);
        phase_n*=2;
        std::vector<std::tuple<T,T>> omega_list; omega_list.resize(itr/2);
        for(size_t i = 0; i < itr/2 ; i ++){
            std::tuple<T,T> omega = omega_func( itr, i);
            {
                // modify omega with phase
                T omr, omi; std::tie(omr,omi) = omega;
                T pmr, pmi; std::tie(pmr,pmi) = phase;
                omega = std::make_tuple<T,T>(omr*pmr-omi*pmi, omr*pmi+omi*pmr);
            }
            omega_list[i] = omega;
        }
        for(size_t g=0;g<groups;g++){
            size_t k = bit_reverse_nbits(g, log2(groups));  
            T omr, omi; std::tie(omr,omi) = omega_list[k];
            for(size_t s=0;s<stride;s++){
                T ar,ai,br,bi,tr,ti;
                LD_C(seq,g*group_len+s,ar,ai);
                LD_C(seq,g*group_len+s+stride,br,bi);
                if(is_inverse_fft)  IBTFL_C(ar,ai,br,bi,omr,omi,tr,ti);
                else                BTFL_C(ar,ai,br,bi,omr,omi,tr,ti);
                ST_C(seq,g*group_len+s,ar,ai);
                ST_C(seq,g*group_len+s+stride,br,bi);
            }
        }
    }
    if(need_final_reverse) bit_reverse_radix2_c(seq, c_length);
    if(is_inverse_fft){
        for(size_t i=0;i<c_length;i++){
            seq[2*i] = seq[2*i]/c_length;
            seq[2*i+1] = seq[2*i+1]/c_length;
        }
    }
}
template<typename T>
static inline void _fft_cooley_tukey_r_mt(T * seq, size_t c_length, bool is_inverse_fft, bool need_final_reverse = true){
    if(c_length == 1) return;
    assert( ( (c_length & (c_length - 1)) == 0 ) && "current only length power of 2");

    std::function<std::tuple<T,T>(size_t,size_t)> omega_func = [](size_t total_n, size_t k){
            T theta = -1*C_2PI*k / total_n;
            return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta)); };

    for(size_t itr = 2; itr<=c_length; itr<<=1){
        size_t stride = c_length/itr;
        size_t groups = itr/2;
        size_t group_len = stride*2;
        std::vector<std::tuple<T,T>> omega_list; omega_list.resize(itr/2);
        for(size_t i = 0; i < itr/2 ; i ++) omega_list[i] = omega_func( itr, i);
        for(size_t g=0;g<groups;g++){
            size_t k = bit_reverse_nbits(g, log2(groups));  
            T omr, omi; std::tie(omr,omi) = omega_list[k];
            for(size_t s=0;s<stride;s++){
                T ar,ai,br,bi,tr,ti;
                LD_C(seq,g*group_len+s,ar,ai);
                LD_C(seq,g*group_len+s+stride,br,bi);
                if(is_inverse_fft)  IBTFL_C(ar,ai,br,bi,omr,omi,tr,ti);
                else                BTFL_C(ar,ai,br,bi,omr,omi,tr,ti);
                ST_C(seq,g*group_len+s,ar,ai);
                ST_C(seq,g*group_len+s+stride,br,bi);
            }
        }
    }
    if(need_final_reverse) bit_reverse_radix2_c(seq, c_length);
    if(is_inverse_fft){
        for(size_t i=0;i<c_length;i++){
            seq[2*i] = seq[2*i]/c_length;
            seq[2*i+1] = seq[2*i+1]/c_length;
        }
    }
}
template<typename T>
static inline void fft_cooley_tukey_r_mt(T * seq, size_t c_length, bool need_final_reverse = true){
    _fft_cooley_tukey_r_mt(seq, c_length, false, need_final_reverse); }
template<typename T>
static inline void ifft_cooley_tukey_r_mt(T * seq, size_t c_length, bool need_final_reverse = true){
    _fft_cooley_tukey_r_mt(seq, c_length, true, need_final_reverse); }

/************************************************************************************/

template<typename T>
void fft8(T * seq, bool need_final_reverse = true){fft_cooley_tukey_r_mt(seq,8,need_final_reverse);}
template<typename T> 
void ifft8(T * seq, bool need_final_reverse = true){ifft_cooley_tukey_r_mt(seq,8,need_final_reverse);}
template<typename T>
void fft8_shifted(T * seq, size_t phase_n, size_t phase_k, bool need_final_reverse = true){
    _fft_cooley_tukey_r_mt_shifted(seq,8,phase_n,phase_k,false,need_final_reverse);}
template<typename T>
void ifft8_shifted(T * seq, size_t phase_n, size_t phase_k, bool need_final_reverse = true){
    _fft_cooley_tukey_r_mt_shifted(seq,8,phase_n,phase_k,true,need_final_reverse);}

template<typename T>
void fft16(T * seq, bool need_final_reverse = true){fft_cooley_tukey_r_mt(seq,16,need_final_reverse);}
template<typename T> 
void ifft16(T * seq, bool need_final_reverse = true){ifft_cooley_tukey_r_mt(seq,16,need_final_reverse);}

template<typename T>
void fft32(T * seq, bool need_final_reverse = true){fft_cooley_tukey_r_mt(seq,32,need_final_reverse);}
template<typename T> 
void ifft32(T * seq, bool need_final_reverse = true){ifft_cooley_tukey_r_mt(seq,32,need_final_reverse);}


#define STRIDE_N_DIVIDE(vec2d,pseq,tlen,ntiles) \
    do{                                         \
        vec2d.resize(ntiles);                   \
        for(size_t i=0;i<ntiles;i++){           \
            vec2d[i].resize(tlen*2);            \
            for(size_t j=0;j<tlen;j++){         \
                vec2d[i][2*j] = pseq[2*(i+j*ntiles)];    \
                vec2d[i][2*j+1] = pseq[2*(i+j*ntiles)+1];\
            }                                   \
        }                                       \
    }while(0)

#define RESTORE_ORIGIN(vec2d,pseq,tlen,ntiles)      \
    do{                                             \
        for(size_t i=0;i<ntiles;i++){               \
            for(size_t j=0;j<tlen;j++){             \
                pseq[2*(i+j*ntiles)] = vec2d[i][2*j];     \
                pseq[2*(i+j*ntiles)+1] = vec2d[i][2*j+1]; \
            }                                       \
        }                                           \
    }while(0)

#define RESTORE_ORIGIN_PERM(vec2d,pseq,tlen,ntiles) \
    do{                                             \
        std::vector<size_t> perm_idx;               \
        bit_reverse_permute(log2(ntiles), perm_idx);\
        for(size_t i=0;i<ntiles;i++){               \
            for(size_t j=0;j<tlen;j++){             \
                pseq[2*(i+j*ntiles)] = vec2d[i][2*j];     \
                pseq[2*(i+j*ntiles)+1] = vec2d[i][2*j+1]; \
            }                                       \
        }                                           \
    }while(0)

//#define RESTORE_PERMUTE

template<typename T>
void fft64(T * seq, bool need_final_reverse = true){
    /*
    * 3+3 division
    * 8xfft8, follow by 8xfft8
    */
    // first step, stride 8, 8 tiles
    std::vector<std::vector<T>> sub_tiles;
    STRIDE_N_DIVIDE(sub_tiles, seq, 8, 8 );

    // do first ffts
    for(size_t i=0;i<8;i++) fft8(sub_tiles[i].data(),false);

    if(need_final_reverse){
#ifdef RESTORE_PERMUTE
#else
        // restore original order
        RESTORE_ORIGIN(sub_tiles, seq, 8, 8 );

        // 2nd step
        std::vector<size_t> idx_arr;
        bit_reverse_permute(log2(8), idx_arr);
        for(size_t i=0;i<8;i++) fft8_shifted(seq+2*8*i, 1<<(3+1), idx_arr[i], false);

        // reorder
        if(need_final_reverse) bit_reverse_radix2_c(seq, 64);
#endif
    }else{
        // restore original order
        RESTORE_ORIGIN(sub_tiles, seq, 8, 8 );

        // 2nd step
        std::vector<size_t> idx_arr;
        bit_reverse_permute(log2(8), idx_arr);
        for(size_t i=0;i<8;i++) fft8_shifted(seq+2*8*i, 1<<(3+1), idx_arr[i], false);
    }
}
template<typename T>
void ifft64(T * seq, bool need_final_reverse = true){
    /*
    * 3+3 division
    * 8xfft8, follow by 8xfft8
    */
    // first step, stride 8, 8 tiles
    std::vector<std::vector<T>> sub_tiles;
    STRIDE_N_DIVIDE(sub_tiles, seq, 8, 8 );

    // do first ffts
    for(size_t i=0;i<8;i++) ifft8(sub_tiles[i].data(),false);

    // restore original order
    RESTORE_ORIGIN(sub_tiles, seq, 8, 8 );

    // 2nd step
    std::vector<size_t> idx_arr;
    bit_reverse_permute(log2(8), idx_arr);
    for(size_t i=0;i<8;i++) ifft8_shifted(seq+2*8*i, 1<<(3+1), idx_arr[i], false);

    // reorder
    if(need_final_reverse) bit_reverse_radix2_c(seq, 64);
}

/************************************************************************************/

template<typename T>
void rand_vec(T *  seq, size_t len){
    static std::random_device rd;   // seed
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(0.0001f, 1.0);
    for(size_t i=0;i<len;i++) seq[i] =  dist(mt);
}
template<typename T>
void copy_vector(const T * src, T *dst, size_t len){
    for(size_t i=0;i<len;i++)   dst[i] = src[i];
}
template<typename T>
int valid_vector(const T* lhs, const T* rhs, size_t len, T delta = (T)0.03){
    int err_cnt = 0;
    for(size_t i = 0;i < len; i++){
        T d = lhs[i]- rhs[i];
#define ABS(x) ((x)>0?(x):-1*(x))
        d = ABS(d);
        if(d > delta){
            std::cout<<" diff at "<<i<<", lhs:"<<lhs[i]<<", rhs:"<<rhs[i]<<", delta:"<<d<<std::endl;
            err_cnt++;
        }
    }
    return err_cnt;
}

void test_fft64(){
    float * seq = (float*)malloc(sizeof(float)*64*2);
    float * seq2 = (float*)malloc(sizeof(float)*64*2);
    printf("fft64\n");
    rand_vec(seq,64*2);
    copy_vector(seq,seq2,64*2);
    fft64(seq);
    fft_cooley_tukey_r_mt(seq2,64);
    valid_vector(seq2,seq,64*2);
    printf("ifft64\n");
    rand_vec(seq,64*2);
    copy_vector(seq,seq2,64*2);
    ifft64(seq);
    ifft_cooley_tukey_r_mt(seq2,64);
    valid_vector(seq2,seq,64*2);
}

int main(){
    test_fft64();
}