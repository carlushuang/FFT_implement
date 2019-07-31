#include <math.h>
#include <vector>
#include <stdint.h>
#include <tuple>
#include <iostream>
#include <assert.h>
#include <functional>
#include <random>
#include <stdlib.h>
#include <limits>

#ifdef USE_FFTW
#include <fftw3.h>
#endif

#define LD_C(vec,idx,r,i) do{r=vec[2*(idx)];i=vec[2*(idx)+1];}while(0)
#define ST_C(vec,idx,r,i) do{vec[2*(idx)]=r;vec[2*(idx)+1]=i;}while(0)

// A=ar+ai*i, B=br+bi*i, omega=omr+omi*i
// A'= A+omega*B = ar+ai*i+(omr+omi*i)*(br+bi*i) = ar+omr*br-omi*bi + (ai+omi*br+omr*bi)*i
// B'= A-omega*B = ar+ai*i-(omr+omi*i)*(br+bi*i) = ar-omr*br+omi*bi + (ai-omr*bi-omi*br)*i
#define BTFL_C(ar,ai,br,bi,omr,omi,tr,ti) do{\
    tr=br*omr-bi*omi;ti=br*omi+bi*omr; \
    br=ar; bi=ai;\
    ar=ar+tr;ai=ai+ti;\
    br=br-tr;bi=bi-ti; } while(0)

// A=ar+ai*i, B=br+bi*i, omega=omr+omi*i
// A'= A+conj(omega)*B = ar+ai*i+(omr-omi*i)*(br+bi*i) = ar+omr*br+omi*bi + (ai-omi*br+omr*bi)*i
// B'= A-conj(omega)*B = ar+ai*i-(omr-omi*i)*(br+bi*i) = ar-omr*br-omi*bi + (ai-omr*bi+omi*br)*i
#define IBTFL_C(ar,ai,br,bi,omr,omi,tr,ti) do{\
    tr=br*omr+bi*omi;ti=-br*omi+bi*omr; \
    br=ar; bi=ai;\
    ar=ar+tr;ai=ai+ti;\
    br=br-tr;bi=bi-ti; } while(0)

#ifndef C_PI
#define C_PI  3.14159265358979323846
#endif
#ifndef C_2PI
#define C_2PI 6.28318530717958647692
#endif

#define PRE_PAD_DATA
#define FFTCONV_USE_CONJ // this is a good mode that all omega use the same function, unified_omega_func_f32
#define FFTCONV_USE_CONJ_NO_ROTATE // this mode, all kernel padding shape is same. we restore output in c2r part
//#define FFTCONV_USE_CONJ_A  // same as FFTCONV_USE_CONJ, but notice, time reverse is fft shift
#define MERGE_2D_NYQUEST_FREQ

#if defined(FFTCONV_USE_CONJ) && defined(FFTCONV_USE_CONJ_A)
#   error "can't both conj and conj_a mode"
#endif

std::tuple<float,float> unified_omega_func_f32(size_t total_n, size_t k){
    float theta = -1*C_2PI*k / total_n;
    return std::make_tuple<float,float>((float)cos(theta), (float)sin(theta));
}

template<typename T>
void dump_vector(const T * vec, size_t len){
    for(size_t i=0;i<len;i++) std::cout<<vec[i]<<", ";
    std::cout<<std::endl;
}
template<typename T>
void dump_vector_2d(const T * vec, size_t width, size_t height){
    for(size_t j=0;j<height;j++){
        for(size_t i=0;i<width;i++){
            std::cout<<vec[j*width+i]<<", ";
        }
        std::cout<<std::endl;
    }
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
template<typename T>
int valid_vector_nrms(const T* pred, const T* ref, size_t len, double tolerance = (double)1e-6)
{
#define RMS_THRESHOLD 1e-6
#ifndef ABS
#define ABS(x)      ((x)>0?(x):-1*(x))
#endif
#ifndef MAX
#define MAX(a,b)    ( (a)>(b)?(a):(b) )
#endif
    // check MIOpen https://github.com/ROCmSoftwarePlatform/MIOpen/blob/master/test/verify.hpp#L167
    // normalized root mean squared error.
    double v, max, nrms;
    v = 0;
    max = std::numeric_limits<double>::min();
    for(size_t i=0;i<len;i++){
        double d = ref[i]-pred[i];
        double m2 = MAX(ABS(ref[i]),ABS(pred[i]));
        v += d*d;
        max = MAX(max,m2);
    }
    nrms = sqrt(v)/(sqrt(len)*max);
    return (nrms<RMS_THRESHOLD)?0:1;
}

template<typename T>
void copy_vector(const T * src, T *dst, size_t len){
    for(size_t i=0;i<len;i++)   dst[i] = src[i];
}
template<typename T>
void rand_vec(T *  seq, size_t len){
    static std::random_device rd;   // seed
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(0.0001f, 1.0);
    for(size_t i=0;i<len;i++) seq[i] =  dist(mt);
}
template<typename T>
T rand_one(){
    static std::random_device rd;   // seed
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(0.0001f, 1.0);
    return dist(mt);
}
// np.fft.fft(...)
// t_seq, f_seq, have length c_length, which should be 2x longer than 2rc algo
template<typename T>
void fft_naive_mt(const T * t_seq, T * f_seq, size_t c_length){
    auto omega_func_n = [](size_t total_n, size_t k, size_t n){
        T theta = -1 * C_2PI*k*n / total_n;
        return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta));
    };
    size_t fft_n = c_length;
    for(size_t k=0;k<fft_n;k++){
        size_t n;
        T omr, omi;
        T fr=(T)0, fi=(T)0;
        for(n=0;n<fft_n;n++){
            std::tie(omr, omi) = omega_func_n(fft_n, k, n);
            fr += t_seq[2*n]*omr-t_seq[2*n+1]*omi;
            fi += t_seq[2*n]*omi+t_seq[2*n+1]*omr;
        }
        f_seq[2*k]=fr;
        f_seq[2*k+1]=fi;
    }
}
template<typename T>
void ifft_naive_mt(T * t_seq, const T * f_seq, size_t c_length){
    auto omega_func_n = [](size_t total_n, size_t k, size_t n){
        T theta = C_2PI*k*n / total_n;
        return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta));
    };
    size_t fft_n = c_length;
    for(size_t k=0;k<fft_n;k++){
        size_t n;
        T omr, omi;
        T fr=(T)0, fi=(T)0;
        for(n=0;n<fft_n;n++){
            std::tie(omr, omi) = omega_func_n(fft_n, k, n);
            fr += f_seq[2*n]*omr-f_seq[2*n+1]*omi;
            fi += f_seq[2*n]*omi+f_seq[2*n+1]*omr;
        }
        t_seq[2*k]=fr;
        t_seq[2*k+1]=fi;
    }
    for(size_t i=0;i<c_length;i++){
        t_seq[2*i] /= c_length;
        t_seq[2*i+1] /= c_length;
    }
}
int bit_reverse_nbits(int v, int nbits){
    int r = 0; int d = nbits-1;
    for(int i=0;i<nbits;i++)
    {   if(v & (1<<i)) r |= 1<<d;  d--; }
    return r;
}
// below function produce  https://oeis.org/A030109
void bit_reverse_permute(size_t radix2_num, std::vector<size_t> &arr)
{
    arr.resize(pow(2,radix2_num));
    arr[0] = 0;
    for(size_t k=0;k<radix2_num;k++){
       size_t last_k_len = pow(2, k);
       size_t last_k;
       for(last_k = 0; last_k < last_k_len; last_k++){
           arr[last_k] = 2*arr[last_k];
           arr[last_k_len+last_k] = arr[last_k]+1;
       }
    }
}
template<typename T>
void bit_reverse_radix2_c(T *vec,size_t c_length){
    assert( ( (c_length & (c_length - 1)) == 0 ) && "must be radix of 2");
    std::vector<size_t> r_idx;
    bit_reverse_permute(log2(c_length), r_idx);
    for(size_t i=0;i<c_length;i++){
        size_t ir = r_idx[i];
        if(i<ir)
            { std::swap(vec[2*i], vec[2*ir]); std::swap(vec[2*i+1], vec[2*ir+1]); }
    }
}
// seq has c_length complex value, 2*c_length value
template<typename T>
void _fft_cooley_tukey_r_mt(T * seq, size_t c_length, bool is_inverse_fft, bool need_final_reverse = true){
    if(c_length == 1) return;
    assert( ( (c_length & (c_length - 1)) == 0 ) && "current only length power of 2");

    std::function<std::tuple<T,T>(size_t,size_t)> omega_func;
#if defined(FFTCONV_USE_CONJ) || defined(FFTCONV_USE_CONJ_A)
    omega_func = [](size_t total_n, size_t k){
            T theta = -1*C_2PI*k / total_n;
            return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta)); };
#else
    if(is_inverse_fft){
        omega_func = [](size_t total_n, size_t k){
            T theta = C_2PI*k / total_n;
            return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta)); };
    }else{
        omega_func = [](size_t total_n, size_t k){
            T theta = -1*C_2PI*k / total_n;
            return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta)); };
    }
#endif

    for(size_t itr = 2; itr<=c_length; itr<<=1){
        size_t stride = c_length/itr;
        size_t groups = itr/2;
        size_t group_len = stride*2;

        std::vector<std::tuple<T,T>> omega_list;   // pre-compute omega, and index to it later
        omega_list.resize(itr/2);
        for(size_t i = 0; i < itr/2 ; i ++){
            omega_list[i] = omega_func( itr, i);
        }
        for(size_t g=0;g<groups;g++){
            size_t k = bit_reverse_nbits(g, log2(groups));  
            T omr, omi;
            std::tie(omr,omi) = omega_list[k];
            for(size_t s=0;s<stride;s++){
                T ar,ai,br,bi,tr,ti;
                LD_C(seq,g*group_len+s,ar,ai);
                LD_C(seq,g*group_len+s+stride,br,bi);
#if defined(FFTCONV_USE_CONJ) || defined(FFTCONV_USE_CONJ_A)
                if(is_inverse_fft)
                    IBTFL_C(ar,ai,br,bi,omr,omi,tr,ti);
                else
                    BTFL_C(ar,ai,br,bi,omr,omi,tr,ti);
#else
                BTFL_C(ar,ai,br,bi,omr,omi,tr,ti);
#endif
                ST_C(seq,g*group_len+s,ar,ai);
                ST_C(seq,g*group_len+s+stride,br,bi);
            }
        }
    }
    if(need_final_reverse)
        bit_reverse_radix2_c(seq, c_length);
    if(is_inverse_fft){
        for(size_t i=0;i<c_length;i++){
            seq[2*i] = seq[2*i]/c_length;
            seq[2*i+1] = seq[2*i+1]/c_length;
        }
    }
}
template<typename T>
void fft_cooley_tukey_r_mt(T * seq, size_t c_length, bool need_final_reverse = true){
    _fft_cooley_tukey_r_mt(seq, c_length, false, need_final_reverse);
}
template<typename T>
void ifft_cooley_tukey_r_mt(T * seq, size_t c_length, bool need_final_reverse = true){
    _fft_cooley_tukey_r_mt(seq, c_length, true, need_final_reverse);
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
*   Gr(k) = 0.5*( Xr(k)*(1-sin) + Xi(k)*cos + Xr(N/2-k)*(1+sin) + Xi(N/2-k)*cos )
*   Gi(k) = 0.5*( Xi(k)*(1-sin) - Xr(k)*cos + Xr(N/2-k)*cos - Xi(N/2-k)(1+sin) )
*
*       -> Gr(0) = Xr(0) + Xi(0)
*       -> Gi(0) = 0
*
*   Gr(N/2) = Xr(0) – Xi(0)
*   Gi(N/2) = 0
*   Gr(N–k) = Gr(k), for k = 1...N/2–1
*   Gi(N–k) = –Gi(k)
*
*   NOTE:
*   r2c->gemm->c2r, then the second half is indeed not needed
*
*
*   NOTE:
*   in 2d r2c, we first vfft r2c for each col, result every N column to N/2+1
*   then do N/2+1 length hfft for each row
*   indeed, we can merge the G(0) and G(N/2) together to G(0), and do hfft, and get back G(0), G(N/2)
*   in this way, we can only do N/2 length hfft for each row.
*
*   Gr(0) = Xr(0) + Xi(0)
*   Gi(0) = 0
*   Gr(N/2) = Xr(0) - Xi(0)
*   Gi(N/2) = 0
*
*   --> the image part of G(0) and G(N/2) is zero, hence we can merge G(0) G(N/2) into signle G(0):
*   Gr(0) = Xr(0) + Xi(0)
*   Gi(0) = Xr(0) - Xi(0)
*
*
*   MERGE_2D_NYQUEST_FREQ
*   then do vfft, and derive back the real fft result of G(0), G(N/2)
*   This problem is equivalent to:
*
*   xa(n) = A+0*j
*   xb(n) = B+0*j
*   x(n) = A+B*j    A, B, is length N vector A(n), B(n), n=0...N-1
*
*   after do the hfft of the merged first row, we already know F.T of x(n) ->X(k)
*   X(k)=sigma((A+B*j)*(cos(@)-sin(@)*j)),  sigma() -> add from 0...N-1. @, theta, is @(k,n)=2*PI*k*n/N
*   X(k)=sigma( A*cos@+B*sin@ +(-A*sin@+B*cos@)*j )
*       =sigma( R0 + I0*j)
*
*   we what to get both:
*   Xa(K) = sigma( A*(cos(@)-sin(@)*j) ) = sigma( A*cos@+(-A*sin@)*j )
*   Xb(K) = sigma( B*(cos(@)-sin(@)*j) ) = sigma( B*cos@+(-B*sin@)*j )
*
*   note that when k item is N-k, and @ has 2*PI period
*   @(N-k,n) = 2*PI*(N-k)*n/N = 2*PI*n-2*PI*k*n/N = -2*PI*k*n/N = -@(k,n)   
*
*   hence:
*   X(N-k)=sigma( A*cos@-B*sin@ +(A*sin@+B*cos@)*j )
*         =sigma( R1 + I1*j)
*
*   So, we can get Xa(k) and Xb(k) from X(k) and X(N-k)
*   Xa(K) = sigma( 0.5*(R0+R1)+0.5*(I0-I1)*j  )
*   Xb(k) = sigma( 0.5*(I0+I1)+0.5*(-R0+R1)*j )
*
*   R0:real part of k-th, X(k)
*   I0:image part of k-th, X(k)
*   R1:real part of (N-k)-th, X(N-k)
*   I1:image part of (N-k)-th, X(N-k)
*
*/
/*
*   Gr(k) = 0.5*( Xr(k)*(1-sin) + Xi(k)*cos + Xr(N/2-k)*(1+sin) + Xi(N/2-k)*cos )
*   Gi(k) = 0.5*( Xi(k)*(1-sin) - Xr(k)*cos + Xr(N/2-k)*cos - Xi(N/2-k)(1+sin) )
*
*   Gr(0) = Xr(0) + Xi(0)
*   Gi(0) = 0
*   Gr(N/2) = Xr(0) - Xi(0)
*   Gi(N/2) = 0
*
*   sin: sin(2*PI*k/N), cos: cos(2*PI*k/N), sin(pi-t) = sin(t), cos(pi-t) = -cos(t)
*
*   Gr(k) = 0.5*( Xr(k)*(1-sin) + Xi(k)*cos + Xr(N/2-k)*(1+sin) + Xi(N/2-k)*cos )
*   Gi(k) = 0.5*( Xi(k)*(1-sin) - Xr(k)*cos + Xr(N/2-k)*cos - Xi(N/2-k)(1+sin) )
*
*   Gr(N/2-k) = 0.5*( Xr(N/2-k)*(1-sin) - Xi(N/2-k)*cos + Xr(k)*(1+sin) - Xi(k)*cos )
*   Gi(N/2-k) = 0.5*( Xi(N/2-k)*(1-sin) + Xr(N/2-k)*cos - Xr(k)*cos - Xi(k)(1+sin) )
*
*  -->
*   Gr(k) = 0.5*( Xr(k)+Xr(N/2-k) - (Xr(k)-Xr(N/2-k))*sin + (Xi(k)+Xi(N/2-k))*cos )
*   Gi(k) = 0.5*( Xi(k)-Xi(N/2-k) - (Xi(k)+Xi(N/2-k))*sin - (Xr(k)-Xr(N/2-k))*cos )
*   Gr(N/2-k) = 0.5*( Xr(k)+Xr(N/2-k)  + (Xr(k)-Xr(N/2-k))*sin - (Xi(k)+Xi(N/2-k))*cos )
*   Gi(N/2-k) = 0.5*( -1*(Xi(k)-Xi(N/2-k)) - (Xi(k)+Xi(N/2-k))*sin - (Xr(k)-Xr(N/2-k))*cos) )
*
*   let:
*    tr0=Xr(k)+Xr(N/2-k), ti0=Xr(k)-Xr(N/2-k), tr1=Xi(k)+Xi(N/2-k), ti1=Xi(k)-Xi(N/2-k)
*
*    Gr(k) = 0.5*(tr0 - ti0*sin + tr1*cos)
*    Gi(k) = 0.5*(ti1 - tr1*sin - ti0*cos)
*    Gr(N/2-k) = 0.5*(tr0 + ti0*sin - tr1*cos)
*    Gi(N/2-k) = 0.5*(-1*ti1 - tr1*sin - ti0*cos)
*
*   for k=N/4, sin(2*PI*k/N)=1, cos(2*PI*k/N)=0
*    Gr(N/4) = Xr(N/4)+Xi(N/4)*cos = Xr(N/4)
*    Gi(N/4) = -1*Xi(N/4)*sin = -1*Xi(N/4)
*/
#define R2C_EPILOG(gr,gi,gnr,gni,s,c,tr0,ti0,tr1,ti1) \
    do{ \
        tr0=gr+gnr; ti0=gr-gnr; tr1=gi+gni; ti1=gi-gni;\
        gr = 0.5*(tr0 - ti0*s + tr1*c); \
        gi = 0.5*(ti1 - tr1*s - ti0*c); \
        gnr = 0.5*(tr0 + ti0*s - tr1*c); \
        gni = -0.5*(ti1 + tr1*s + ti0*c);\
    }while(0)

/* t_seq, f_seq all length long.
* t_seq is length real
* f_seq is complex, if merge_nyquist_freq == true: length value, length/2 complex value.
*                   if merge_nyquist_freq == false: length+2 value, length/2+1 complex value.
* indeed, the output f_seq should be length/2+1 complex value, for the 0-th and length/2-th value are all real only complex number.(Nyquist frequency)
* if merge_nyquist_freq == true, we merge the real part of 0-th, length/2-th real part to 0-th real/image part, this can save space. but make multi-dim fft complicated
*/
template<typename T>
void fft_r2c_mt(const T* t_seq, T * f_seq, size_t length, bool merge_nyquist_freq=false){
    if(length == 1) return;
    assert( ((length & (length - 1)) == 0 ) && "current only length power of 2");
    T tmp;
    auto omega_func = [](size_t total_n, size_t k){
        T theta = C_2PI*k / total_n;
        return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta));
    };

    std::vector<std::tuple<T,T>> omega_list;
    omega_list.resize(length/2);
    for(size_t i=0;i<length/2;i++){
        omega_list[i] = omega_func(length,i);
    }

    std::vector<size_t> brev;
    bit_reverse_permute(log2(length/2), brev);
    for(size_t i=0;i<length;i++){
        f_seq[i] = t_seq[i];
    }
    fft_cooley_tukey_r_mt(f_seq, length/2, false);

    tmp = f_seq[0];
    f_seq[0] = f_seq[0]+f_seq[1];
    if(merge_nyquist_freq)
        f_seq[1] = tmp-f_seq[1];        // merge Gr(N/2) = Xr(0) - Xi(0)
    else{
        f_seq[length] = tmp-f_seq[1];
        f_seq[1] = 0;
        f_seq[length+1] = 0;
    }

    if(length == 2) return;
    for(size_t i=0;i<(length/4-1);i++){
        size_t idx = i+1;
        size_t brev_idx = brev[idx];
        size_t brev_idx_r = brev[length/2-idx];
        T gr,gi,gnr,gni,s,c,tr0,ti0,tr1,ti1;
        std::tie(c,s) = omega_list[idx];
        LD_C(f_seq,brev_idx,gr,gi);
        LD_C(f_seq,brev_idx_r,gnr,gni);
        R2C_EPILOG(gr,gi,gnr,gni,s,c,tr0,ti0,tr1,ti1);
        if(brev_idx != idx ){
            std::swap( brev[brev_idx] , brev[idx]);
            std::swap( f_seq[2*brev_idx], f_seq[2*idx] );
            std::swap( f_seq[2*brev_idx+1], f_seq[2*idx+1] );
        }
        if(brev_idx_r != (length/2-idx)){
            std::swap(brev[brev_idx_r], brev[length/2-idx]);
            std::swap(f_seq[2*brev_idx_r], f_seq[2*(length/2-idx)]);
            std::swap(f_seq[2*brev_idx_r+1], f_seq[2*(length/2-idx)+1]);
        }
        ST_C(f_seq,idx,gr,gi);
        ST_C(f_seq,length/2-idx,gnr,gni);
    }
    if(length/4){
        f_seq[2*(length/4)] = f_seq[2*(length/4)];
        f_seq[2*(length/4)+1] = -1*f_seq[2*(length/4)+1];
    }
}
/*
* t_seq, f_seq all length long.
* t_seq is seq_h*seq_w real
* f_seq is complex, if merge_nyquist_freq == true: (seq_h/2)*(2*seq_w) value, (seq_h/2)*seq_w complex value.
*                   if merge_nyquist_freq == false: (seq_h/2+1)*(2*seq_w) value, (seq_h/2+1)*seq_w complex value.
* indeed, 2d r2c merge_nyquist_freq can't be true, otherwise original information will be corrupt
* But we can merge it while do horizontal fft
*/
template<typename T>
void fft2d_r2c_mt(const T* t_seq, T * f_seq, size_t seq_w, size_t seq_h){
    bool h_merge_nyquist_freq=
#ifdef MERGE_2D_NYQUEST_FREQ
        true;
#else
        false;
#endif
    // vertical
    T * vt = new T[seq_h];
    T * vf = new T[h_merge_nyquist_freq?seq_h:(seq_h+2)];
    size_t v_len = h_merge_nyquist_freq?seq_h:(seq_h+2);
    for(size_t w=0;w<seq_w;w++){
        for(size_t h=0;h<seq_h;h++){
            vt[h] = t_seq[h*seq_w+w];
        }
        fft_r2c_mt(vt, vf, seq_h, h_merge_nyquist_freq);

        for(size_t h=0;h<v_len/2;h++){
            f_seq[h*2*seq_w+2*w] = vf[2*h];
            f_seq[h*2*seq_w+2*w+1] = vf[2*h+1];
        }
    }
    delete [] vt;
    delete [] vf;

#if 0
    for(size_t h=0;h<v_len/2;h++)
        fft_cooley_tukey_r_mt(f_seq+h*2*seq_w, seq_w);
#endif
#if 1
    auto omega_func = [](size_t total_n, size_t k){
        T theta = -C_2PI*k / total_n;
        return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta));
    };

    // horizontal
    T * h_even = new T[seq_w];
    T * h_odd  = new T[seq_w];
    for(size_t h=0;h<v_len/2;h++){
        for(size_t w=0;w<seq_w/2;w++){
            h_even[2*w]     = f_seq[h*2*seq_w+4*w+0];
            h_even[2*w+1]   = f_seq[h*2*seq_w+4*w+1];
            h_odd[2*w]      = f_seq[h*2*seq_w+4*w+2];
            h_odd[2*w+1]    = f_seq[h*2*seq_w+4*w+3];
        }
        fft_cooley_tukey_r_mt(h_even, seq_w/2);
        fft_cooley_tukey_r_mt(h_odd, seq_w/2);

        for(size_t w=0;w<seq_w/2;w++){
            T c,s;
            std::tie(c,s) = omega_func(seq_w, w);
            // even:er+ei*i, odd:or+oi*i, omega:wr+wi*i
            //
            // er+ei*i+(or+oi*i)*(wr+wi*i)
            // er+or*wr-oi*wi + (ei+or*wi+oi*wr)i
            //
            // er+ei*i-(or+oi*i)*(wr+wi*i)
            // er-or*wr+oi*wi + (ei-or*wi-oi*wr)*i
            //
            f_seq[h*2*seq_w+2*w] = h_even[2*w]+h_odd[2*w]*c-h_odd[2*w+1]*s;
            f_seq[h*2*seq_w+2*w+1] = h_even[2*w+1]+h_odd[2*w]*s+h_odd[2*w+1]*c;

            f_seq[h*2*seq_w+seq_w+2*w] = h_even[2*w]-h_odd[2*w]*c+h_odd[2*w+1]*s;
            f_seq[h*2*seq_w+seq_w+2*w+1] = h_even[2*w+1]-h_odd[2*w]*s-h_odd[2*w+1]*c;
        }
    }
    if(h_merge_nyquist_freq){
        /*   Xa(K) = sigma( 0.5*(R0+R1)+0.5*(I0-I1)*j  )
        *   Xb(k) = sigma( 0.5*(I0+I1)+0.5*(-R0+R1)*j )
        *
        *   R0:real part of k-th, X(k)
        *   I0:image part of k-th, X(k)
        *   R1:real part of (N-k)-th, X(N-k)
        *   I1:image part of (N-k)-th, X(N-k)
        */

        // point 0
        f_seq[0] = f_seq[0];
        f_seq[(seq_h/2)*2*seq_w] = f_seq[1];
        f_seq[1] = 0;
        f_seq[(seq_h/2)*2*seq_w+1] = 0;

        // point N/2
        //float rr,ii;
        //rr = f_seq[seq_w];
        //ii = f_seq[seq_w+1];
        //f_seq[seq_w] = rr;
        //f_seq[seq_w+1] = 0;
        //f_seq[(seq_h/2)*2*seq_w+seq_w] = ii;
        //f_seq[(seq_h/2)*2*seq_w+seq_w+1] = 0;

        for(size_t w=1;w<=seq_w/2;w++){
            float r0,r1,i0,i1;
            r0 = f_seq[2*w];
            i0 = f_seq[2*w+1];
            r1 = f_seq[2*(seq_w-w)];
            i1 = f_seq[2*(seq_w-w)+1];
            // row 0
            f_seq[2*w] = 0.5*(r0+r1);
            f_seq[2*w+1] = 0.5*(i0-i1);
            f_seq[2*(seq_w-w)] = 0.5*(r1+r0);
            f_seq[2*(seq_w-w)+1] = 0.5*(i1-i0);

            // row seq_h/2+1
            f_seq[(seq_h/2)*2*seq_w+2*w] = 0.5*(i0+i1);
            f_seq[(seq_h/2)*2*seq_w+2*w+1] = 0.5*(-r0+r1);
            f_seq[(seq_h/2)*2*seq_w+2*(seq_w-w)] = 0.5*(i1+i0);
            f_seq[(seq_h/2)*2*seq_w+2*(seq_w-w)+1] = 0.5*(-r1+r0);
        }
    }
    delete [] h_even;
    delete [] h_odd;
#endif
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
*   G(N/2) = G(0)
*
*   IA : complex conjugate of A
*   IB : complex conjugate of B
*   IAr(k) = 0.5*(1.0-sin(2*PI*k/N))
*   IAi(k) = 0.5*(1*cos(2*PI*k/N))
*   IBr(k) = 0.5*(1+sin(2*PI*k/N))
*   IBi(k) = 0.5*(-1*cos(2*PI*k/N))
*               k=0...N/2-1
*
*   Xr(k) = 0.5*( Gr(k)*(1-sin) – Gi(k)*cos + Gr(N/2–k)*(1+sin) - Gi(N/2–k)*cos )
*   Xi(k) = 0.5*( Gi(k)*(1-sin) + Gr(k)*cos - Gr(N/2–k)*cos – Gi(N/2–k)*(1+sin) )
*                for k = 0...N/2–1
*
*   for k, N/2-k, the sin/cos has following pattern:
*   sin: sin(2*PI*k/N), cos: cos(2*PI*k/N), sin(pi-t) = sin(t), cos(pi-t) = -cos(t)
*
*   Xr(k) = 0.5*( Gr(k)*(1-sin) – Gi(k)*cos + Gr(N/2–k)*(1+sin) - Gi(N/2–k)*cos )
*   Xi(k) = 0.5*( Gi(k)*(1-sin) + Gr(k)*cos - Gr(N/2–k)*cos – Gi(N/2–k)*(1+sin) )
*
*   Xr(N/2-k) = 0.5*( Gr(N/2-k)*(1-sin) + Gi(N/2-k)*cos + Gr(k)*(1+sin) + Gi(k)*cos )
*   Xi(N/2-k) = 0.5*( Gi(N/2-k)*(1-sin) - Gr(N/2-k)*cos + Gr(k)*cos - Gi(k)*(1+sin) )
*
*  -->
*   Xr(k) = 0.5*( Gr(k)+Gr(N/2–k) - sin*(Gr(k)-Gr(N/2–k)) - cos*(Gi(k)+Gi(N/2–k)) )
*   Xi(k) = 0.5*( Gi(k)-Gi(N/2–k) - sin*(Gi(k)+Gi(N/2–k)) + cos*(Gr(k)-Gr(N/2–k)) )
*   Xr(N/2-k) = 0.5*( Gr(k)+Gr(N/2-k) + sin*(Gr(k)-Gr(N/2-k)) + cos*(Gi(k)+Gi(N/2-k)) )
*   Xi(N/2-k) = 0.5*( -(Gi(k)-Gi(N/2-k)) -sin(Gi(k)+Gi(N/2-k)) + cos(Gr(k)-Gr(N/2-k)) )
*
*   let:
*    sr0=Gr(k)+Gr(N/2–k), si0=Gr(k)-Gr(N/2–k), sr1=Gi(k)+Gi(N/2-k), si1=Gi(k)-Gi(N/2-k)
*
*    Xr(k) = 0.5*(sr0 - si0*sin - sr1*cos)
*    Xi(k) = 0.5*(si1 - sr1*sin + si0*cos)
*    Xr(N/2-k) = 0.5*(sr0 + si0*sin + sr1*cos)
*    Xi(N/2-k) = 0.5*(-1*si1 - sr1*sin + si0*cos)
*
*    Xr(0) = 0.5*(Gr(0)+Gr(N/2) - Gi(0)-Gi(N/2)) = 0.5*( Gr(0)-Gi(N/2) - (Gi(0)-Gr(N/2)) )
*                                                = 0.5*( Gr(0) + Gr(N/2))
*    Xi(0) = 0.5*(Gi(0)-Gi(N/2) + Gr(0)-Gr(N/2)) = 0.5*( Gr(0)-Gi(N/2) + Gi(0) -Gr(N/2)  )
*                                                = 0.5*( Gr(0) - Gr(N/2))
*
*   for k=N/4, sin(2*PI*k/N)=1, cos(2*PI*k/N)=0
*    Xr(N/4) = Gr(N/4)
*    Xi(N/4) = -1*Gi(N/4)
*
*   [w conj case], theta = -2*PI*k/N
*   IAr(k) = 0.5*(1.0+sin(-2*PI*k/N))
*   IAi(k) = 0.5*(1*cos(-2*PI*k/N))
*   IBr(k) = 0.5*(1-sin(-2*PI*k/N))
*   IBi(k) = 0.5*(-1*cos(-2*PI*k/N))
*               k=0...N/2-1
*    Xr(k) = Gr(k)IAr(k) – Gi(k)IAi(k) + Gr(N/2–k)IBr(k) + Gi(N/2–k)IBi(k)
*    Xi(k) = Gi(k)IAr(k) + Gr(k)IAi(k) + Gr(N/2–k)IBi(k) – Gi(N/2–k)IBr(k)
*
*    Xr(k) = 0.5*( Gr(k)*(1+sin) – Gi(k)*cos + Gr(N/2–k)*(1-sin) - Gi(N/2–k)*cos )
*    Xi(k) = 0.5*( Gi(k)*(1+sin) + Gr(k)*cos - Gr(N/2–k)*cos – Gi(N/2–k)*(1-sin) )
*
*    Xr(k) = 0.5*( Gr(k)+Gr(N/2–k) + sin*(Gr(k)-Gr(N/2–k)) - cos*(Gi(k)+Gi(N/2–k)) )
*    Xi(k) = 0.5*( Gi(k)-Gi(N/2–k) + sin*(Gi(k)+Gi(N/2–k)) + cos*(Gr(k)-Gr(N/2–k)) )
*
*    Xr(N/2-k) = 0.5*( Gr(k)+Gr(N/2–k) - sin*(Gr(k)-Gr(N/2–k)) + cos*(Gi(k)+Gi(N/2–k)) )
*    Xi(N/2-k) = 0.5*( -Gi(k)+Gi(N/2–k) + sin*(Gi(k)+Gi(N/2–k)) + cos*(Gr(k)-Gr(N/2–k)) )
*   let:
*    sr0=Gr(k)+Gr(N/2–k), si0=Gr(k)-Gr(N/2–k), sr1=Gi(k)+Gi(N/2-k), si1=Gi(k)-Gi(N/2-k)
*
*    Xr(k) = 0.5*(sr0 + si0*sin - sr1*cos)
*    Xi(k) = 0.5*(si1 + sr1*sin + si0*cos)
*    Xr(N/2-k) = 0.5*(sr0 - si0*sin + sr1*cos)
*    Xi(N/2-k) = 0.5*(-1*si1 + sr1*sin + si0*cos)
*
*    Xr(0)=0.5*(Gr(0)+Gr(N/2) - Gi(0) - Gi(N/2)  ) =0.5*(Gr(0) + Gr(N/2))
*    Xi(0)=0.5*(Gi(0)-Gi(N/2) + Gr(0) - Gr(N/2)  ) =0.5*(Gr(0) - Gr(N/2))
*
*   for k=N/4, sin(-2*PI*k/N)=-1, cos(-2*PI*k/N)=0
*    Xr(N/4) = 0.5*(sr0-si0) = Gr(N/4)
*    Xi(N/4) = 0.5*(si1-sr1) = -Gi(N/4)
*
*
* MERGE_2D_NYQUEST_FREQ
*
* a(n) = ar(n)+ai(n)*j  ifft==>  A(K) = Ar(k)+0*j
* b(n) = br(n)+bi(n)*j  ifft==>  B(K) = Br(k)+0*j
*
* q(n) = qr(n)+qi(n)*j  ifft==>  Q(k) = Ar(k)+Br(k)*j
*
* W=e^(-2*PI/N), W_k_n=e^(-2*k*n*PI/N)
* W=cos(@)+sin(@)*j, @ = -2*PI/N
*
* sigma_n(W_k_n) = sigma_n(conj(W_k_n))= N*theta(k mod N)
* theta(x) = (x==0)?1:0
*
* A(k) = 1/N*sigma_n(a(n)*conj(W_k_n))
* B(k) = 1/N*sigma_n(b(n)*conj(W_k_n))
*
* q(n) = sigma_k( Q(k)*W_k_n ) = sigma_k(
*    (  1/N*sigma_l(a(l)*conj(W_k_l))  +  1/N*sigma_l(b(l)*conj(W_k_l))*j  )*W_k_n )
*    =1/N*sigma_l(  a(l)*sigma_k(W_k_(n-l))  ) + 1/N*sigma_l(  b(l)*sigma_k(W_k_(n-l)) )*j
*    =1/N*sigma_l(  a(l)*N*theta(n-l mod N)  ) + 1/N*sigma_l(  b(l)*N*theta(n-l mod N) )*j
*
*    =a(l) + b(l)*j
*    =ar(n)+ai(n)*j + (br(n)+bi(n)*j)*j
*    =ar(n)-bi(n) + (ai(n)+br(n))*j
*
*/
/* when convolution case, suppose a, b is data, c, d is filter
* c(n) = cr(n)+ci(n)*j  ifft==>  C(K) = Cr(k)+0*j
* d(n) = dr(n)+di(n)*j  ifft==>  D(K) = Dr(k)+0*j
*
* r(n) = rr(n)+ri(n)*j  ifft==>  R(k) = Cr(k)+Dr(k)*j
*      = cr(n)-di(n) + (ci(n)+dr(n))*j
*
*
* x(n) = a(n)*conj(c(n)) = ar(n)*cr(n)+ai(n)*ci(n)+(-ar(n)*ci(n)+ai(n)*cr(n))*j
* y(n) = b(n)*conj(d(n)) = br(n)*dr(n)+bi(n)*di(n)+(-br(n)*di(n)+bi(n)*dr(n))*j
*
* X(k) = 1/N*sigma_n( a(n)*conj(c(n))*conj(W_k_n) )
* Y(k) = 1/N*sigma_n( b(n)*conj(d(n))*conj(W_k_n) )
*
* X(k) = 1/N*sigma_n( a(n)*conj(c(n))*conj(W_k_n) )
*      = 1/N*sigma_n( sigma_l(A(l)*W_l_n) * conj(sigma_l(C(l)*W_l_n)) * conj(W_k_n) )
*      = 1/N*sigma_n( sigma_l(A(l)*W_l_n) * conj(W_k_n) ) * sigma_n( conj(sigma_l(C(l)*W_l_n)) )
*      = 1/N*sigma_l( A(l)*sigma_n(W_(k-l)_n)) ) * sigma_n( conj(sigma_l(C(l)*W_l_n)) )
*      = 1/N*sigma_l( A(l)*N*theta(k-l mod N)  ) * sigma_n( conj(sigma_l(C(l)*W_l_n)) )
*      = A(k) * sigma_n(conj(sigma_l(C(l)*W_l_n)) )
*      = A(k) * sigma_n(conj(c(n)))
*
* Y(k) = B(k) * sigma_n(conj(d(n)))
*
* since sigma_n(conj(c(n))), sigma_n(conj(d(n))), the image part is zero (recall r2c hermitian symmertry)
* the new X(k)/Y(k) is also real only, hence safe to reuse that in 1d c2r
*
*/

#define C2R_EPILOG(xr,xi,xnr,xni,s,c,sr0,si0,sr1,si1)   \
    do{                                                 \
        sr0=xr+xnr; si0=xr-xnr; sr1=xi+xni; si1=xi-xni; \
        xr = 0.5*(sr0 - si0*s - sr1*c);                 \
        xi = 0.5*(si1 - sr1*s + si0*c);                 \
        xnr = 0.5*(sr0 + si0*s + sr1*c);                \
        xni = 0.5*(-1*si1 - sr1*s + si0*c);             \
    }while(0)

#define IC2R_EPILOG(xr,xi,xnr,xni,s,c,sr0,si0,sr1,si1)  \
    do{                                                 \
        sr0=xr+xnr; si0=xr-xnr; sr1=xi+xni; si1=xi-xni; \
        xr = 0.5*(sr0 + si0*s - sr1*c);                 \
        xi = 0.5*(si1 + sr1*s + si0*c);                 \
        xnr = 0.5*(sr0 - si0*s + sr1*c);                \
        xni = 0.5*(-1*si1 + sr1*s + si0*c);             \
    }while(0)


/*
* t_seq, f_seq all length long.
* t_seq is length real
* f_seq is complex, if merge_nyquist_freq == true: length value, length/2 complex value.
*                   if merge_nyquist_freq == false: length+2 value, length/2+1 complex value.
*/
template<typename T>
void ifft_c2r_mt(T* t_seq, const T * f_seq, size_t length, bool merge_nyquist_freq=false){
    // the 0-th and length/2-th complex number, image part must be zero, same as fftw
    if(length == 1) return;
    assert( ((length & (length - 1)) == 0 ) && "current only length power of 2");

#if defined(FFTCONV_USE_CONJ) || defined(FFTCONV_USE_CONJ_A)
    auto omega_func = [](size_t total_n, size_t k){
        T theta = -1*C_2PI*k / total_n;
        return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta));
    };
#else
    auto omega_func = [](size_t total_n, size_t k){
        T theta = C_2PI*k / total_n;
        return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta));
    };
#endif

    std::vector<std::tuple<T,T>> omega_list;
    omega_list.resize(length/2);
    for(size_t i=0;i<length/2;i++){
        omega_list[i] = omega_func(length,i);
    }

    if(length == 2) return;

#if defined(FFTCONV_USE_CONJ) || defined(FFTCONV_USE_CONJ_A)
    if(!merge_nyquist_freq){
        t_seq[0] = 0.5*(f_seq[0]+f_seq[length]);
        t_seq[1] = 0.5*(f_seq[0]-f_seq[length]);
    }else{
        t_seq[0] = 0.5*(f_seq[0]+f_seq[1]);
        t_seq[1] = 0.5*(f_seq[0]-f_seq[1]);
    }
#else
    // Xr(0) = 0.5*( Gr(0)-Gi(N/2) - (Gi(0)-Gr(N/2)) )
    // Xi(0) = 0.5*( Gr(0)-Gi(N/2) + Gi(0) -Gr(N/2)  )
    // Here we assume 0-th and length/2-th complex number only have real part, other wise it's not c2r
    if(!merge_nyquist_freq){
        t_seq[0] = 0.5*(f_seq[0]+f_seq[length]);
        t_seq[1] = 0.5*(f_seq[0]-f_seq[length]);
    }else{
        t_seq[0] = 0.5*(f_seq[0]+f_seq[1]);
        t_seq[1] = 0.5*(f_seq[0]-f_seq[1]);
    }
#endif

    for(size_t i=1;i<=(length/4-1);i++){
        T xr,xi,xnr,xni,s,c,sr0,si0,sr1,si1;
        std::tie(c,s) = omega_list[i];

        LD_C(f_seq,i,xr,xi);
        LD_C(f_seq,length/2-i,xnr,xni);
#if defined(FFTCONV_USE_CONJ) || defined(FFTCONV_USE_CONJ_A)
        IC2R_EPILOG(xr,xi,xnr,xni,s,c,sr0,si0,sr1,si1);
#else
        C2R_EPILOG(xr,xi,xnr,xni,s,c,sr0,si0,sr1,si1);
#endif
        ST_C(t_seq,i,xr,xi);
        ST_C(t_seq,length/2-i,xnr,xni);
    }
    if(length/4){
        t_seq[2*(length/4)] = f_seq[2*(length/4)];
        t_seq[2*(length/4)+1] = -1*f_seq[2*(length/4)+1];
    }
    ifft_cooley_tukey_r_mt(t_seq, length/2, true);
}
/*
* t_seq is seq_h*seq_w real
* f_seq is complex, if merge_nyquist_freq == true: (seq_h/2)*(2*seq_w) value, (seq_h/2)*seq_w complex value.
*                   if merge_nyquist_freq == false: (seq_h/2+1)*(2*seq_w) value, (seq_h/2+1)*seq_w complex value.
* indeed, 2d c2r merge_nyquist_freq can't be true, otherwise original information will be corrupt
*/
template<typename T>
void ifft2d_c2r_mt(T* t_seq, const T * f_seq, size_t seq_w, size_t seq_h){
    bool h_merge_nyquist_freq = 
#ifdef MERGE_2D_NYQUEST_FREQ
        true;
#else
        false;
#endif
    size_t v_len = h_merge_nyquist_freq?seq_h:(seq_h+2);
    T * seq = new T[v_len*seq_w];
    float * f_seq_first_row = NULL;
    if(h_merge_nyquist_freq){
        f_seq_first_row = new float[2*seq_w];
        for(size_t w=0;w<seq_w;w++){
            //   ar(n)+ai(n)*j + (br(n)+bi(n)*j)*j
            //   ar(n)-bi(n) + (ai(n)+br(n))*j
            f_seq_first_row[2*w] = f_seq[2*w]-f_seq[(seq_h/2)*2*seq_w+2*w+1 ];
            f_seq_first_row[2*w+1] = f_seq[2*w+1]+f_seq[(seq_h/2)*2*seq_w+2*w ];
        }
    }

    // horizontal
#if 0
    for(size_t h=0;h<v_len/2;h++){
        if(h_merge_nyquist_freq && h==0){
            for(size_t w=0;w<2*seq_w;w++){
                seq[h*2*seq_w+w] = f_seq_first_row[w];
            }
        }else{
            for(size_t w=0;w<2*seq_w;w++){
                seq[h*2*seq_w+w] = f_seq[h*2*seq_w+w];
            }
        }
        ifft_cooley_tukey_r_mt(seq+h*2*seq_w, seq_w, true);
    }
#endif
#if 1
    T * h_even = new T[seq_w];
    T * h_odd  = new T[seq_w];
#if defined(FFTCONV_USE_CONJ) || defined(FFTCONV_USE_CONJ_A)
    auto omega_func = [](size_t total_n, size_t k){
        T theta = -1*C_2PI*k / total_n;
        return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta));
    };
#else
    auto omega_func = [](size_t total_n, size_t k){
        T theta = C_2PI*k / total_n;
        return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta));
    };
#endif
    
    for(size_t h=0;h<v_len/2;h++){
        if( h_merge_nyquist_freq && h==0){
            for(size_t w=0;w<seq_w/2;w++){
                h_even[2*w]     = f_seq_first_row[4*w+0];
                h_even[2*w+1]   = f_seq_first_row[4*w+1];
                h_odd[2*w]      = f_seq_first_row[4*w+2];
                h_odd[2*w+1]    = f_seq_first_row[4*w+3];
            }
        }
        else{
            for(size_t w=0;w<seq_w/2;w++){
                h_even[2*w]     = f_seq[h*2*seq_w+4*w+0];
                h_even[2*w+1]   = f_seq[h*2*seq_w+4*w+1];
                h_odd[2*w]      = f_seq[h*2*seq_w+4*w+2];
                h_odd[2*w+1]    = f_seq[h*2*seq_w+4*w+3];
            }
        }
        ifft_cooley_tukey_r_mt(h_even, seq_w/2, true);
        ifft_cooley_tukey_r_mt(h_odd, seq_w/2, true);

        for(size_t w=0;w<seq_w/2;w++){
            T c,s;
            std::tie(c,s) = omega_func(seq_w, w);
#if defined(FFTCONV_USE_CONJ) || defined(FFTCONV_USE_CONJ_A)
            seq[h*2*seq_w+2*w] = (h_even[2*w]+h_odd[2*w]*c+h_odd[2*w+1]*s)/2;
            seq[h*2*seq_w+2*w+1] = (h_even[2*w+1]-h_odd[2*w]*s+h_odd[2*w+1]*c)/2;
            seq[h*2*seq_w+seq_w+2*w] = (h_even[2*w]-h_odd[2*w]*c-h_odd[2*w+1]*s)/2;
            seq[h*2*seq_w+seq_w+2*w+1] = (h_even[2*w+1]+h_odd[2*w]*s-h_odd[2*w+1]*c)/2;
#else
            // even:er+ei*i, odd:or+oi*i, omega:wr+wi*i
            //
            // er+ei*i+(or+oi*i)*(wr+wi*i)
            // er+or*wr-oi*wi + (ei+or*wi+oi*wr)i
            //
            // er+ei*i-(or+oi*i)*(wr+wi*i)
            // er-or*wr+oi*wi + (ei-or*wi-oi*wr)*i
            //
            seq[h*2*seq_w+2*w] = (h_even[2*w]+h_odd[2*w]*c-h_odd[2*w+1]*s)/2;
            seq[h*2*seq_w+2*w+1] = (h_even[2*w+1]+h_odd[2*w]*s+h_odd[2*w+1]*c)/2;
            seq[h*2*seq_w+seq_w+2*w] = (h_even[2*w]-h_odd[2*w]*c+h_odd[2*w+1]*s)/2;
            seq[h*2*seq_w+seq_w+2*w+1] = (h_even[2*w+1]-h_odd[2*w]*s-h_odd[2*w+1]*c)/2;
#endif
        }
    }
    delete [] h_even;
    delete [] h_odd;
#endif
    if(h_merge_nyquist_freq){
        delete [] f_seq_first_row;
        //for(size_t w=0;w<seq_w/2;w++){
        //    seq[2*w] = seq[2*w];
        //    seq[(seq_h/2)*2*seq_w+2*w ] = seq[2*w+1];
        //    seq[2*w+1] = 0;
        //    seq[(seq_h/2)*2*seq_w+2*w +1] = 0;
        //}
    }
    T * vf = new T[v_len];
    T * vt = new T[seq_h];
    // vertical
    for(size_t w=0;w<seq_w;w++){
        for(size_t h=0;h<v_len/2;h++){
            vf[2*h] = seq[h*2*seq_w+2*w];
            vf[2*h+1] = seq[h*2*seq_w+2*w+1];
        }
        ifft_c2r_mt(vt, vf, seq_h, h_merge_nyquist_freq);
        for(size_t h=0;h<seq_h;h++){
            t_seq[h*seq_w+w] = vt[h];
        }
    }
    delete [] seq;
    delete [] vf;
    delete [] vt;
}

// t_seq is seq_h * 2*seq_w value, seq_h * seq_2 complex
// f_seq is seq_h * 2*seq_w value, seq_h * seq_2 complex
template<typename T>
void fft2d_naive(const T* t_seq, T * f_seq, size_t seq_w, size_t seq_h){
    // vertical
    T * v_seq = new T[seq_h*2];
    for(size_t w=0;w<seq_w;w++){
        for(size_t h=0;h<seq_h;h++){
            v_seq[2*h] = t_seq[h*2*seq_w+2*w];
            v_seq[2*h+1] = t_seq[h*2*seq_w+2*w+1];
        }
        fft_cooley_tukey_r_mt(v_seq, seq_h);
        for(size_t h=0;h<seq_h;h++){
            f_seq[h*2*seq_w+2*w] = v_seq[2*h];
            f_seq[h*2*seq_w+2*w+1] = v_seq[2*h+1];
        }
    }
    // horizontal
    for(size_t h=0;h<seq_h;h++){
        fft_cooley_tukey_r_mt(f_seq+h*2*seq_w,seq_w);
    }
    delete [] v_seq;
}
template<typename T>
void ifft2d_naive(T* t_seq, const T * f_seq, size_t seq_w, size_t seq_h){
    // horizontal
    for(size_t h=0;h<seq_h;h++){
        for(size_t w=0;w<seq_w;w++){
            t_seq[h*2*seq_w+2*w] = f_seq[h*2*seq_w+2*w];
            t_seq[h*2*seq_w+2*w+1] = f_seq[h*2*seq_w+2*w+1];
        }
        ifft_cooley_tukey_r_mt(t_seq+h*2*seq_w,seq_w);
    }

    // vertical
    T * v_seq = new T[seq_h*2];
    for(size_t w=0;w<seq_w;w++){
        for(size_t h=0;h<seq_h;h++){
            v_seq[2*h] = t_seq[h*2*seq_w+2*w];
            v_seq[2*h+1] = t_seq[h*2*seq_w+2*w+1];
        }
        ifft_cooley_tukey_r_mt(v_seq, seq_h);
        for(size_t h=0;h<seq_h;h++){
            t_seq[h*2*seq_w+2*w] = v_seq[2*h];
            t_seq[h*2*seq_w+2*w+1] = v_seq[2*h+1];
        }
    }
    delete [] v_seq;
}

// this is ML/AI convolution, not that in signal process.
template<typename T>
void convolve2d_naive(const T* data, size_t data_w, size_t data_h,
    const T* filter, size_t filter_w, size_t filter_h,
    size_t pad_h, size_t pad_w,
    T* dst, bool correlation = false)
{
    assert((filter_w-1)>=pad_w);
    assert((filter_h-1)>=pad_h);
    // NOTE: in ML/AI, padding is different from signal.
    // in signal process, padding is acctually data_size+filter_size-1, to get full DFT
    // following fomula use ML/AI definition
    size_t dst_h = data_h + 2*pad_h - filter_h + 1;
    size_t dst_w = data_w + 2*pad_w - filter_w + 1;
    size_t i,j,ki,kj;

    std::vector<T> _ff;
    const T * f = filter;

    if(!correlation){
        _ff.resize(filter_w*filter_h);
        for(size_t ii=0;ii<filter_w*filter_h;ii++){
            _ff[ii] = filter[filter_w*filter_h-1-ii];
        }
        f = _ff.data();
    }

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

// this is ML/AI convolution, not that in signal process.
template<typename T>
void convolve2d_fft_mt(const T* data, size_t data_w, size_t data_h,
    const T* filter, size_t filter_w, size_t filter_h,
    size_t pad_h, size_t pad_w,
    T* dst, bool correlation = false)
{
    assert((filter_w-1)>=pad_w);
    assert((filter_h-1)>=pad_h);
    size_t dst_h = data_h + 2*pad_h - filter_h + 1;
    size_t dst_w = data_w + 2*pad_w - filter_w + 1;
    // ML/AI, output size is data_size+2*pad-filter_size+1
    // But in signal process, convolve means data_size+filter_size-1 sequency is convolved
    // Hence we must padding to signal process FFT size (data_size+filter_size-1)
    // Otherwise, the output is shifted.
    size_t seq_pad_h = (size_t)std::pow(2, std::ceil(std::log2(data_h + filter_h-1)));
    size_t seq_pad_w = (size_t)std::pow(2, std::ceil(std::log2(data_w + filter_w-1)));
    //printf("pad to %lux%lu\n",seq_pad_h,seq_pad_w);
    //bool merge_nyquist_freq = false;    // indeed, this can't be true
    //size_t fft_h = merge_nyquist_freq?(seq_pad_h/2):(seq_pad_h/2+1);
    size_t fft_h = seq_pad_h/2+1;
    size_t fft_w = 2*seq_pad_w;

    T * seq_data    = new T[seq_pad_h*seq_pad_w];
    T * seq_filter  = new T[seq_pad_h*seq_pad_w];
    T * fft_data    = new T[fft_h*fft_w];
    T * fft_filter  = new T[fft_h*fft_w];
    T * fft_out     = new T[fft_h*fft_w];
    T * dst_pad     = new T[seq_pad_h*seq_pad_w];

    for(size_t i=0;i<seq_pad_h*seq_pad_w;i++){
        seq_data[i] = (T)0;
        seq_filter[i] = (T)0;
    }

    for(size_t j=0;j<data_h;j++){
        for(size_t i=0;i<data_w;i++){
#ifdef PRE_PAD_DATA
            // PAD HERE! hence the shift is only filter_size-1 from signal process
            seq_data[(j+pad_h)*seq_pad_w+i+pad_w] = data[j*data_w+i];
#else
            seq_data[j*seq_pad_w+i] = data[j*data_w+i];
#endif
        }
    }


    if(correlation){
#if defined(FFTCONV_USE_CONJ) || defined(FFTCONV_USE_CONJ_A)
#   ifdef FFTCONV_USE_CONJ_NO_ROTATE
        for(size_t j=0;j<filter_h;j++){
            for(size_t i=0;i<filter_w;i++){
                seq_filter[j*seq_pad_w+i] = filter[j*filter_w+i]; // no reverse needed
            }
        }
#   else
        /*
        * www.claysturner.com/dsp/timereversal.pdf
        *  corr(a, b) = ifft(fft(a_and_zeros) * conj(fft(b_and_zeros))) [1]
        *  But in DFT, can not pad b_and_zeros with filter. There must be a rotation
        * 
        *            origin             to be correlate (b_and_zeros)
        *  1d: [0, 1, 2, 3, _, _, _, _] -> [3, _, _, _, _, 0, 1, 2]    _ means padding zero
        * 
        *                                               rotate right : seq_pad_w-filter_w+1, rotate left: filter_w-1
        *                                               rotate down  : seq_pad_h-filter_h+1, rotate up  : filter_h-1
        *  2d: [0, 1, 2, _, _, _, _, _] -> [8, _, _, _, _, _, 6, 7]
        *      [3, 4, 5, _, _, _, _, _]    [_, _, _, _, _, _, _, _]
        *      [6, 7, 8, _, _, _, _, _]    [_, _, _, _, _, _, _, _]
        *      [_, _, _, _, _, _, _, _]    [_, _, _, _, _, _, _, _] filter
        *      [_, _, _, _, _, _, _, _]    [_, _, _, _, _, _, _, _]
        *      [_, _, _, _, _, _, _, _]    [_, _, _, _, _, _, _, _]
        *      [_, _, _, _, _, _, _, _]    [2, _, _, _, _, _, 0, 1]
        *      [_, _, _, _, _, _, _, _]    [5, _, _, _, _, _, 3, 4]
        * 
        *                            correlate 
        *
        *  2d: [d, d, d, d, d, d, _, _]    [d, d, d, d, d, d, _, _]
        *      [d, d, d, d, d, d, _, _]    [d, d, d, d, d, d, _, _]
        *      [d, d, d, d, d, d, _, _]    [d, d, d, d, d, d, _, _]
        *      [d, d, d, d, d, d, _, _]    [d, d, d, d, d, d, _, _] data
        *      [d, d, d, d, d, d, _, _]    [d, d, d, d, d, d, _, _]
        *      [d, d, d, d, d, d, _, _]    [d, d, d, d, d, d, _, _]
        *      [_, _, _, _, _, _, _, _]    [_, _, _, _, _, _, _, _]
        *      [_, _, _, _, _, _, _, _]    [_, _, _, _, _, _, _, _]
        *
        *                               ||
        * 
        *  2d: [7, 8, 9, a, b, -, -, 6]    [-, -, -, -, -, -, -, -]
        *      [d, e, f, g, h, -, -, c]    [-, 0, 1, 2, 3, 4, 5, -]
        *      [j, k, l, m, n, -, -, i]    [-, 6, 7, 8, 9, a, b, -]
        *      [p, q, r, s, t, -, -, o]    [-, c, d, e, f, g, h, -] out, - means don't care
        *      [v, w, x, y, z, -, -, u]    [-, i, j, k, l, m, n, -]
        *      [-, -, -, -, -, -, -, -]    [-, o, p, q, r, s, t, -]
        *      [-, -, -, -, -, -, -, -]    [-, u, v, w, x, y, z, -]
        *      [1, 2, 3, 4, 5, -, -, 0]    [-, -, -, -, -, -, -, -]
        */
       for(size_t j=0;j<filter_h;j++){
            for(size_t i=0;i<filter_w;i++){
                size_t dj = (seq_pad_h-filter_h+1+j)%seq_pad_h;
                size_t di = (seq_pad_w-filter_w+1+i)%seq_pad_w;
                seq_filter[dj*seq_pad_w+di] = filter[j*filter_w+i];
            }
        }
#   endif // FFTCONV_USE_CONJ_NO_ROTATE
#else
        for(size_t j=0;j<filter_h;j++){
            for(size_t i=0;i<filter_w;i++){
                seq_filter[j*seq_pad_w+i] = filter[(filter_h-1-j)*filter_w+filter_w-1-i]; // reverse!
            }
        }
#endif // FFTCONV_USE_CONJ
    }else{
        for(size_t j=0;j<filter_h;j++){
            for(size_t i=0;i<filter_w;i++){
                seq_filter[j*seq_pad_w+i] = filter[j*filter_w+i];
            }
        }
    }

    // 1: fft data, fft filter
    fft2d_r2c_mt(seq_data, fft_data, seq_pad_w, seq_pad_h);
    fft2d_r2c_mt(seq_filter, fft_filter, seq_pad_w, seq_pad_h);
    //printf("----------------------- data T\n");
    //dump_vector_2d(seq_data, seq_pad_w, seq_pad_h);
    //printf("----------------------- data F\n");
    //dump_vector_2d(fft_data, fft_w, fft_h);
    //printf("----------------------- filter F\n");
    //dump_vector_2d(fft_filter, fft_w, fft_h);
#if 0
    {
        // FFTW
        fftw_plan p;
        double * in;
        fftw_complex * out;
        in = (double*)fftw_malloc(sizeof(double)*seq_pad_w*seq_pad_h);
        float * fs = new float[seq_pad_h*2*(seq_pad_w/2+1)];
        for(size_t i=0;i<seq_pad_w*seq_pad_h;i++){
            in[i] = seq_data[i];
        }
        out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*  (seq_pad_h)*(seq_pad_w/2+1)    );
        // fftw compute ifft unnormalized. need divide by N
        p=fftw_plan_dft_r2c_2d(seq_pad_w,seq_pad_h, in, out, FFTW_ESTIMATE);
        fftw_execute(p);
        for(size_t j=0;j<seq_pad_h;j++){
            for(size_t i=0;i<(seq_pad_w/2+1);i++){
               fs[j*2*(seq_pad_w/2+1)+2*i] = out[j*(seq_pad_w/2+1)+i][0];
               fs[j*2*(seq_pad_w/2+1)+2*i+1] = out[j*(seq_pad_w/2+1)+i][1];
            }
        }
        printf("----------------------- data F fftw\n");
        dump_vector_2d(fs, 2*(seq_pad_w/2+1), seq_pad_h);
        printf("--------------------------");
        fftw_destroy_plan(p);
        fftw_free(out);
        fftw_free(in);
        free(fs);
    }
#endif

    // 2: element wise multiply
    // if merge_nyquist_freq is true, 2d fft can not get orignal result
    for(size_t j=0;j<fft_h;j++){
        for(size_t i=0;i<fft_w/2;i++){
#ifdef FFTCONV_USE_CONJ
            if(correlation){
                // data*conj(filter)-> (dr+di*i) * (fr-fi*i) -> dr*fr+di*fi+(-dr*fi+di*fr)*i
                fft_out[j*fft_w+2*i]   = fft_data[j*fft_w+2*i]*fft_filter[j*fft_w+2*i] + fft_data[j*fft_w+2*i+1]*fft_filter[j*fft_w+2*i+1];
                fft_out[j*fft_w+2*i+1] = -1*fft_data[j*fft_w+2*i]*fft_filter[j*fft_w+2*i+1] + fft_data[j*fft_w+2*i+1]*fft_filter[j*fft_w+2*i];
            }else
#elif defined(FFTCONV_USE_CONJ_A)
            if(correlation){
                // conj(data)*filter-> (dr-di*i) * (fr+fi*i) -> dr*fr+di*fi+(dr*fi-di*fr)*i
                fft_out[j*fft_w+2*i]   = fft_data[j*fft_w+2*i]*fft_filter[j*fft_w+2*i] + fft_data[j*fft_w+2*i+1]*fft_filter[j*fft_w+2*i+1];
                fft_out[j*fft_w+2*i+1] = fft_data[j*fft_w+2*i]*fft_filter[j*fft_w+2*i+1] - fft_data[j*fft_w+2*i+1]*fft_filter[j*fft_w+2*i];
            }else
#endif
            {
            fft_out[j*fft_w+2*i]   = fft_data[j*fft_w+2*i]*fft_filter[j*fft_w+2*i] - fft_data[j*fft_w+2*i+1]*fft_filter[j*fft_w+2*i+1];
            fft_out[j*fft_w+2*i+1] = fft_data[j*fft_w+2*i]*fft_filter[j*fft_w+2*i+1] + fft_data[j*fft_w+2*i+1]*fft_filter[j*fft_w+2*i];
            }

        }
    }
    //printf("----------------------- out F\n");
    //dump_vector_2d(fft_out, fft_w, fft_h);


    // 3: ifft output
    ifft2d_c2r_mt(dst_pad, fft_out, seq_pad_w, seq_pad_h);
    //printf("----------------------------+++++\n");
    //dump_vector_2d(dst_pad,seq_pad_w,seq_pad_h);
    //printf("----------------------------+++++\n");

    // This is the shift value from signal process to ML/AI meaninig.
    // e.g if filter_size=3, pad=2, then singal and ML/AI has the same meaning.
#ifdef PRE_PAD_DATA
    // PAD HERE! hence the shift is only filter_size-1 from signal process
    size_t shift_h = filter_h-1;
    size_t shift_w = filter_w-1;
#ifdef FFTCONV_USE_CONJ
#   ifdef FFTCONV_USE_CONJ_NO_ROTATE
    (void)shift_h;
    (void)shift_w;
    for(size_t j=0;j<dst_h;j++){
        for(size_t i=0;i<dst_w;i++){
            //size_t sj=(seq_pad_h-filter_h+1+j+shift_h)%seq_pad_h;
            //size_t si=(seq_pad_w-filter_w+1+i+shift_w)%seq_pad_w;
            //size_t sj=(seq_pad_h+j)%seq_pad_h;
            //size_t si=(seq_pad_w+i)%seq_pad_w;
            size_t sj = j;
            size_t si = i;
            // NOTICE: must do shift to get back what ML/AI needed. see PRE_PAD_DATA to check 2 padding method
            dst[j*dst_w+i] = dst_pad[sj*seq_pad_w+si];
        }
    }
#   else
    for(size_t j=0;j<dst_h;j++){
        for(size_t i=0;i<dst_w;i++){
            // NOTICE: must do shift to get back what ML/AI needed. see PRE_PAD_DATA to check 2 padding method
            dst[j*dst_w+i] = dst_pad[(j+shift_h)*seq_pad_w+i+shift_w];
        }
    }
#   endif // FFTCONV_USE_CONJ_NO_ROTATE
#elif defined(FFTCONV_USE_CONJ_A)
#   ifdef FFTCONV_USE_CONJ_NO_ROTATE
    (void)shift_h;
    (void)shift_w;
    for(size_t j=0;j<dst_h;j++){
        for(size_t i=0;i<dst_w;i++){
            size_t sj = (seq_pad_h-j)%seq_pad_h;
            size_t si = (seq_pad_w-i)%seq_pad_w;
            dst[j*dst_w+i] = dst_pad[sj*seq_pad_w+si];
        }
    }
#   else
    assert(0 && "_____ not implemented _____");
#   endif
#else
    assert(0 && "_____ not implemented _____");
#endif
#else
    size_t shift_h = filter_h-1-pad_h;
    size_t shift_w = filter_w-1-pad_w;
#if 0
#ifdef FFTCONV_USE_CONJ_NO_ROTATE
    T * dst_pad_rotated = new T[seq_pad_w*seq_pad_h];
    for(size_t i=0;i<seq_pad_w*seq_pad_h;i++){
        dst_pad_rotated[i] = dst_pad[i];
    }
    for(size_t j=0;j<seq_pad_h;j++){
        for(size_t i=0;i<seq_pad_w;i++){
            size_t sj=(seq_pad_h-filter_h+1+j)%seq_pad_h;
            size_t si=(seq_pad_w-filter_w+1+i)%seq_pad_w;
            dst_pad[j*seq_pad_w+i] = dst_pad_rotated[sj*seq_pad_w+si];
        }
    }
    delete [] dst_pad_rotated;
#endif
    for(size_t j=0;j<dst_h;j++){
        for(size_t i=0;i<dst_w;i++){
            // NOTICE: must do shift to get back what ML/AI needed. see PRE_PAD_DATA to check 2 padding method
            dst[j*dst_w+i] = dst_pad[(j+shift_h)*seq_pad_w+i+shift_w];
        }
    }
#endif
#if 1
#ifdef FFTCONV_USE_CONJ_NO_ROTATE
    for(size_t j=0;j<dst_h;j++){
        for(size_t i=0;i<dst_w;i++){
            //size_t sj=(seq_pad_h-filter_h+1+j+shift_h)%seq_pad_h;
            //size_t si=(seq_pad_w-filter_w+1+i+shift_w)%seq_pad_w;
            size_t sj=(seq_pad_h-pad_h+j)%seq_pad_h;
            size_t si=(seq_pad_w-pad_w+i)%seq_pad_w;
            // NOTICE: must do shift to get back what ML/AI needed. see PRE_PAD_DATA to check 2 padding method
            dst[j*dst_w+i] = dst_pad[sj*seq_pad_w+si];
        }
    }
#else
    for(size_t j=0;j<dst_h;j++){
        for(size_t i=0;i<dst_w;i++){
            // NOTICE: must do shift to get back what ML/AI needed. see PRE_PAD_DATA to check 2 padding method
            dst[j*dst_w+i] = dst_pad[(j+shift_h)*seq_pad_w+i+shift_w];
        }
    }
#endif // FFTCONV_USE_CONJ_NO_ROTATE
#endif
#endif

    
    delete []  seq_data;
    delete []  seq_filter;
    delete []  fft_data;
    delete []  fft_filter;
    delete []  fft_out;
    delete []  dst_pad;
}

/*********************************************************************************/
#define FFT_LEN 8

void test_fft(){
    printf("[%s]\n",__func__);
    float ts[2*FFT_LEN];
    float fs[2*FFT_LEN];
    //rand_vec(ts,2*FFT_LEN);
    for(size_t i=0;i<2*FFT_LEN;i++) ts[i] = i;

    fft_naive_mt(ts,fs,FFT_LEN);
    //dump_vector(ts,2*FFT_LEN);
    //dump_vector(fs,2*FFT_LEN);
    fft_cooley_tukey_r_mt(ts, FFT_LEN);
    //dump_vector(ts,2*FFT_LEN);
    int err=valid_vector(fs,ts,2*FFT_LEN);
    printf("%s\n",err==0?"ok":"fail");
}
void test_ifft(){
    printf("[%s]\n",__func__);
    float ts[2*FFT_LEN];
    float fs[2*FFT_LEN];
    //rand_vec(ts,2*FFT_LEN);
    for(size_t i=0;i<2*FFT_LEN;i++) fs[i] = i;

    ifft_naive_mt(ts,fs,FFT_LEN);
    //dump_vector(ts,2*FFT_LEN);
    //dump_vector(fs,2*FFT_LEN);
    ifft_cooley_tukey_r_mt(fs, FFT_LEN);
    //dump_vector(fs,2*FFT_LEN);
    int err=valid_vector(fs,ts,2*FFT_LEN);
    printf("%s\n",err==0?"ok":"fail");
}
void test_fft_r2c(){
    printf("[%s]\n",__func__);
    float ts[FFT_LEN];
    float ts2[2*FFT_LEN];
    float fs[FFT_LEN];
    float fs2[2*FFT_LEN];
    rand_vec(ts,FFT_LEN);
    fft_r2c_mt(ts,fs,FFT_LEN);
#ifdef USE_FFTW
    (void)ts2;
    {
        fftw_plan p;
        fftw_complex *out;
        double * in;
        in = (double*)fftw_malloc(sizeof(double)*FFT_LEN);
        for(size_t i=0;i<FFT_LEN;i++){
            in[i] = ts[i];
        }
        out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(FFT_LEN/2+1));
        p=fftw_plan_dft_r2c_1d(FFT_LEN,in,out, FFTW_ESTIMATE);
        fftw_execute(p);
        for(size_t i=0;i<FFT_LEN/2;i++){
            fs2[2*i] = out[i][0];
            fs2[2*i+1] = out[i][1];
        }
        fftw_destroy_plan(p);
        fftw_free(out);
        fftw_free(in);
    }
#else
    {
        for(size_t i=0;i<FFT_LEN;i++){
            ts2[2*i] = ts[i];
            ts2[2*i+1] = 0;
        }
        fft_naive_mt(ts2,fs2,FFT_LEN);
        //fs2[1] = fs2[FFT_LEN];
    }
#endif
    int err=valid_vector(fs,fs2,FFT_LEN);
    printf("%s\n",err==0?"ok":"fail");
    //dump_vector(fs,FFT_LEN);
    //dump_vector(fs2,2*FFT_LEN);
}
void test_ifft_c2r(){
    printf("[%s]\n",__func__);
    const bool merg_ny = true;
    float ts[FFT_LEN];
    float ts2[2*FFT_LEN];
    float fs[FFT_LEN+2];
    float fs2[2*FFT_LEN];
    rand_vec(fs,FFT_LEN+2);
    //for(size_t ii=0;ii<(FFT_LEN+2);ii++) fs[ii] = ii+1;
    //fs[1]=fs[FFT_LEN+1]=0;
    if(merg_ny)
        fs[1] = fs[FFT_LEN];
    ifft_c2r_mt(ts,fs,FFT_LEN,merg_ny);
#ifdef USE_FFTW
    (void)fs2;
    {
        fftw_plan p;
        double * out;
        fftw_complex * in;
        in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(FFT_LEN/2+1));
        for(size_t i=0;i<(FFT_LEN/2+1);i++){
            in[i][0] = fs[2*i];
            in[i][1] = fs[2*i+1];
        }
        out = (double*)fftw_malloc(sizeof(double)*FFT_LEN);
        // fftw compute ifft unnormalized. need divide by N
        p=fftw_plan_dft_c2r_1d(FFT_LEN,in,out, FFTW_ESTIMATE);
        fftw_execute(p);
        for(size_t i=0;i<FFT_LEN;i++){
            ts2[i] = out[i]/FFT_LEN;
        }
        fftw_destroy_plan(p);
        fftw_free(out);
        fftw_free(in);
    }
#else
    {
        // restore the hermitian structure of the full complex value, other wise can not get the correct value   
        //     b) for second half:
        //      Gr(N/2) = Xr(0) - Xi(0),    real - imag
        //      Gi(N/2) = 0
        //      G(N-k) = G*(k), k:1...N/2-1
        fs2[0] = fs[0];
        fs2[1] = 0;
        for(size_t i=1;i<FFT_LEN/2;i++){
            fs2[2*i] = fs[2*i];
            fs2[2*i+1] = fs[2*i+1];
        }
        fs2[FFT_LEN] = fs[FFT_LEN];
        fs2[FFT_LEN+1] = 0;
        for(size_t i=1;i<FFT_LEN/2;i++){
            fs2[2*FFT_LEN-2*i] = fs2[2*i];
            fs2[2*FFT_LEN-2*i+1] = -fs2[2*i+1];
        }
        //fs2[1]=fs2[FFT_LEN+1]=fs2[2*FFT_LEN-1]=0;

        ifft_naive_mt(ts2,fs2,FFT_LEN);
        for(size_t i=0;i<FFT_LEN;i++){
            ts2[i] = ts2[2*i];
        }
    }
#endif
    int err=valid_vector(ts,ts2,FFT_LEN);
    printf("%s\n",err==0?"ok":"fail");
}
void test_fft2d_r2c(){
    printf("[test fft2d r2c]\n");
    //bool merge_nyquist_freq=false;
    float fs[(FFT_LEN/2+1)*(2*FFT_LEN)];
    float fs2[(FFT_LEN/2+1)*(2*FFT_LEN)];
    float ts[FFT_LEN*FFT_LEN];
    //rand_vec(ts,FFT_LEN);
    for(size_t i=0;i<FFT_LEN*FFT_LEN;i++) ts[i] = 2*i+i*i*(i<5?0.2:(i<10?0.1:0.04));
    fft2d_r2c_mt(ts, fs, FFT_LEN, FFT_LEN);
    dump_vector_2d(fs, 2*FFT_LEN,  FFT_LEN/2+1 );
    printf("-------------------------\n");
#ifdef USE_FFTW
    {
        //http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html
        // fftw r2c 2d is first hfft, then vfft, different from us.
        fftw_plan p;
        double * in;
        fftw_complex * out;
        in = (double*)fftw_malloc(sizeof(double)*FFT_LEN*FFT_LEN);
        for(size_t i=0;i<FFT_LEN*FFT_LEN;i++){
            in[i] = ts[i];
        }
        out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*  ( (FFT_LEN/2+1)*FFT_LEN )     );
        // fftw compute ifft unnormalized. need divide by N
        p=fftw_plan_dft_r2c_2d(FFT_LEN,FFT_LEN, in, out, FFTW_ESTIMATE);
        fftw_execute(p);
        for(size_t j=0;j<FFT_LEN;j++){
            for(size_t i=0;i<(FFT_LEN/2+1);i++){
               fs2[j*2*(FFT_LEN/2+1)+2*i] = out[j*(FFT_LEN/2+1)+i][0];
               fs2[j*2*(FFT_LEN/2+1)+2*i+1] = out[j*(FFT_LEN/2+1)+i][1];
            }
        }
        dump_vector_2d(fs2, 2*(FFT_LEN/2+1), FFT_LEN);
        fftw_destroy_plan(p);
        fftw_free(out);
        fftw_free(in);
    }
#else
    printf("... not implemented reference\n");
#endif
    {
        printf("------------------------- 2d not r2c\n");
        float ts_naive[FFT_LEN*2*FFT_LEN];
        for(size_t j=0;j<FFT_LEN;j++){
            for(size_t i=0;i<FFT_LEN;i++){
                ts_naive[j*2*FFT_LEN+2*i] = ts[j*FFT_LEN+i];
                ts_naive[j*2*FFT_LEN+2*i+1] = 0;
            }
        }
        float fs_naive[FFT_LEN*2*FFT_LEN];
        fft2d_naive(ts_naive, fs_naive, FFT_LEN, FFT_LEN);
        dump_vector_2d(fs_naive, 2*FFT_LEN, FFT_LEN);
    }
}
void test_ifft2d_c2r(){
    printf("[%s]\n",__func__);
    //bool merge_nyquist_freq=false;
    float fs[(FFT_LEN/2+1)*(2*FFT_LEN)];
    float fs2[(FFT_LEN/2+1)*(2*FFT_LEN)];
    float ts[FFT_LEN*FFT_LEN];
    float ts2[FFT_LEN*FFT_LEN];
    //rand_vec(ts,FFT_LEN);
    //for(size_t i=0;i<(FFT_LEN/2+1)*(2*FFT_LEN);i++) fs[i] = 2*i+i*i*(i<5?0.2:(i<10?0.1:0.04));
    {
        // it's hard to generate symmetric 2s sequency, so I use fft
        float ts_naive[FFT_LEN*2*FFT_LEN];
        for(size_t j=0;j<FFT_LEN;j++){
            for(size_t i=0;i<FFT_LEN;i++){
                //ts_naive[j*2*FFT_LEN+2*i] = 2*j+1+i*i*(i<5?0.5:(i<10?0.8:0.2));
                ts_naive[j*2*FFT_LEN+2*i] = rand_one<float>();
                ts_naive[j*2*FFT_LEN+2*i+1] = 0;
            }
        }
        float fs_naive[FFT_LEN*2*FFT_LEN];
        fft2d_naive(ts_naive, fs_naive, FFT_LEN, FFT_LEN);
        printf("------------------------- fft-ed value:\n");
        dump_vector_2d(fs_naive, 2*FFT_LEN, FFT_LEN);
        

        //
        for(size_t j=0;j<FFT_LEN/2+1;j++){
            for(size_t i=0;i<FFT_LEN;i++){
                fs[j*2*FFT_LEN+2*i] =  fs_naive[j*2*FFT_LEN+2*i];
                fs[j*2*FFT_LEN+2*i+1] =  fs_naive[j*2*FFT_LEN+2*i+1];
            }
        }

        for(size_t j=0;j<FFT_LEN;j++){
            for(size_t i=0;i<(FFT_LEN/2+1);i++){
                fs2[j*2*(FFT_LEN/2+1)+2*i] =  fs_naive[j*2*FFT_LEN+2*i];
                fs2[j*2*(FFT_LEN/2+1)+2*i+1] =  fs_naive[j*2*FFT_LEN+2*i+1];
            }
        }

        ifft2d_naive(ts_naive, fs_naive, FFT_LEN, FFT_LEN);
        printf("------------------------- ifft naive:\n");
        dump_vector_2d(ts_naive, 2*FFT_LEN, FFT_LEN);
        
    }
    ifft2d_c2r_mt(ts, fs, FFT_LEN, FFT_LEN);
    printf("------------------------- ifft c2r mt:\n");
    dump_vector_2d(ts, FFT_LEN,  FFT_LEN );
    
#ifdef USE_FFTW
    {
        //http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html
        // fftw c2r 2d is first hfft, then vfft, different from us.
        fftw_plan p;
        fftw_complex * in;
        double * out;
        in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(FFT_LEN/2+1)*FFT_LEN);
        for(size_t j=0;j<FFT_LEN;j++){
            for(size_t i=0;i<(FFT_LEN/2+1);i++){
                in[j*(FFT_LEN/2+1)+i][0] = fs2[j*2*(FFT_LEN/2+1)+2*i];
                in[j*(FFT_LEN/2+1)+i][1] = fs2[j*2*(FFT_LEN/2+1)+2*i+1];
            }
        }
        out = (double*)fftw_malloc(sizeof(double)* FFT_LEN*FFT_LEN  );
        // fftw compute ifft unnormalized. need divide by N
        p=fftw_plan_dft_c2r_2d(FFT_LEN,FFT_LEN, in, out, FFTW_ESTIMATE);
        fftw_execute(p);

        for(size_t i=0;i<FFT_LEN*FFT_LEN;i++){
            ts2[i] = out[i]/(FFT_LEN*FFT_LEN);
        }
        printf("------------------------- ifft c2r fftw:\n");
        dump_vector_2d(ts2, FFT_LEN, FFT_LEN);
        fftw_destroy_plan(p);
        fftw_free(out);
        fftw_free(in);
    }
#else
    printf("... not implemented reference\n");
#endif
}
void test_convolve_2d(){
    printf("[%s]\n",__func__);
    struct {
        size_t w;
        size_t p;
        size_t f;
    }cfg[] =
    //   w   p  f
    {   {14, 3, 7},
        {3,  0, 1},
        {8,  1, 3},
        {31, 1, 3},
        {19, 2, 5},
        {19, 0, 5},
        {55, 3, 7},
        {81, 4, 11},
        {9,  2, 3},
        {10, 0, 4},
        {32, 0, 32},
    };
    
    for(size_t i=0;i<sizeof(cfg)/sizeof(cfg[0]);i++){
        size_t data_wh=cfg[i].w;
        size_t pad_wh=cfg[i].p;
        size_t filter_wh=cfg[i].f;

        size_t out_wh =  data_wh + 2*pad_wh - filter_wh + 1;

        float * data = new float[data_wh*data_wh];
        float * filter = new float[filter_wh*filter_wh];
        float * out = new float[out_wh*out_wh];
        float * out_mt = new float[out_wh*out_wh];

        printf("size:%-3lu, pad:%-2lu, f:%-2lu, ",data_wh,pad_wh,filter_wh);

        rand_vec(data,data_wh*data_wh);
        rand_vec(filter,filter_wh*filter_wh);
        //for(size_t i=0;i<data_wh*data_wh;i++) data[i] = i;
        //for(size_t i=0;i<filter_wh*filter_wh;i++) filter[i] = i;

        convolve2d_naive(data, data_wh, data_wh, filter, filter_wh, filter_wh, pad_wh, pad_wh, out, true);
        convolve2d_fft_mt(data, data_wh, data_wh, filter, filter_wh, filter_wh, pad_wh, pad_wh, out_mt, true);

        //printf("-------------------- out\n");
        //dump_vector_2d(out,out_wh, out_wh);
        //printf("-------------------- out_mt\n");
        //dump_vector_2d(out_mt,out_wh, out_wh);

        //int err=valid_vector(out,out_mt,out_wh*out_wh);
        int err=valid_vector_nrms(out,out_mt,out_wh*out_wh);
        printf("%s\n",err==0?"ok":"fail");

        delete [] data;
        delete [] filter;
        delete [] out;
        delete [] out_mt;
    }
    
}

int main(){
    //test_fft();
    //test_ifft();
    //test_fft_r2c();
    //test_ifft_c2r();
    //test_fft2d_r2c();
    //test_ifft2d_c2r();
    test_convolve_2d();
}