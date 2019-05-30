#include <math.h>
#include <vector>
#include <stdint.h>
#include <tuple>
#include <iostream>
#include <assert.h>
#include <functional>
#include <random>
#include <stdlib.h>

#define LD_C(vec,idx,r,i) do{r=vec[2*(idx)];i=vec[2*(idx)+1];}while(0)
#define ST_C(vec,idx,r,i) do{vec[2*(idx)]=r;vec[2*(idx)+1]=i;}while(0)
#if 0
#define BTFL_C(ar,ai,br,bi,omr,omi,tr,ti) do{\
    tr=ar;ti=ai; \
    ar=ar+br*omr-bi*omi;ai=ai+br*omi+bi*omr;\
    br=tr-br*omr+bi*omi;bi=ti-br*omi-bi*omr; } while(0)
#endif
#define BTFL_C(ar,ai,br,bi,omr,omi,tr,ti) do{\
    tr=br*omr-bi*omi;ti=br*omi+bi*omr; \
    br=ar; bi=ai;\
    ar=ar+tr;ai=ai+ti;\
    br=br-tr;bi=bi-ti; } while(0)

#ifndef C_PI
#define C_PI  3.14159265358979323846
#define C_2PI 6.28318530717958647692
#endif
template<typename T>
void dump_vector(const T * vec, size_t len){
    for(size_t i=0;i<len;i++) std::cout<<vec[i]<<", ";
    std::cout<<std::endl;
}
template<typename T>
int valid_vector(const T* lhs, const T* rhs, size_t len, T delta = (T)0.001){
    int err_cnt = 0;
    for(size_t i = 0;i < len; i++){
        T d = lhs[i]- rhs[i];
        d = abs(d);
        if(d > delta){
            std::cout<<" diff at "<<i<<", lhs:"<<lhs[i]<<", rhs:"<<rhs[i]<<std::endl;
            err_cnt++;
        }
    }
    return err_cnt;
}
template<typename T>
void copy_vector(const T * src, T *dst, size_t len){
    for(size_t i=0;i<len;i++)   dst[i] = src[i];
}
template<typename T>
void rand_vec(T *  seq, size_t len){
    static std::random_device rd;   // seed
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(-2.0, 2.0);
    for(size_t i=0;i<len;i++) seq[i] =  dist(mt);
}
// np.fft.fft(...)
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
template<typename T>
void _fft_cooley_tukey_r_mt(T * seq, size_t c_length, bool is_inverse_fft){
    if(c_length == 1) return;
    assert( ( (c_length & (c_length - 1)) == 0 ) && "current only length power of 2");

    std::function<std::tuple<T,T>(size_t,size_t)> omega_func;
    if(is_inverse_fft){
        omega_func = [](size_t total_n, size_t k){
            T theta = C_2PI*k / total_n;
            return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta)); };
    }else{
        omega_func = [](size_t total_n, size_t k){
            T theta = -1*C_2PI*k / total_n;
            return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta)); };
    }

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
                BTFL_C(ar,ai,br,bi,omr,omi,tr,ti);
                ST_C(seq,g*group_len+s,ar,ai);
                ST_C(seq,g*group_len+s+stride,br,bi);
            }
        }
    }

    bit_reverse_radix2_c(seq, c_length);
    if(is_inverse_fft){
        for(size_t i=0;i<c_length;i++){
            seq[2*i] = seq[2*i]/c_length;
            seq[2*i+1] = seq[2*i+1]/c_length;
        }
    }
}
template<typename T>
void fft_cooley_tukey_r_mt(T * seq, size_t c_length){
    _fft_cooley_tukey_r_mt(seq, c_length, false);
}
template<typename T>
void ifft_cooley_tukey_r_mt(T * seq, size_t c_length){
    _fft_cooley_tukey_r_mt(seq, c_length, true);
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
*   sin: sin(2*PI*k/N), cos: cos(2*PI*k/N), sin(pi-t) = sin(t), cos(pi-t) = cos(t)
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
*   for k=N/4
*    Gr(N/4) = Xr(N/4)+Xi(N/4)*cos
*    Gi(N/4) = -1*Xi(N/4)*sin
*/
#define R2C_EPILOG(gr,gi,gnr,gni,s,c,tr0,ti0,tr1,ti1) \
    do{ \
        tr0=gr+gnr; ti0=gr-gnr; tr1=gi+gni; ti1=gi-gni;\
        gr = 0.5*(tr0 - ti0*s + tr1*c); \
        gi = 0.5*(ti1 - tr1*s - ti0*c); \
        gnr = 0.5*(tr0 + ti0*s - tr1*c); \
        gni = 0.5*(-1*ti1 - tr1*s - ti0*c);\
    }while(0)
template<typename T>
void fft_r2c_mt(const T* t_seq, T * f_seq, size_t length){
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

    for(size_t i=0;i<length;i++){
        f_seq[i] = t_seq[i];
    }
    fft_cooley_tukey_r_mt(f_seq, length/2);

    tmp = f_seq[0];
    f_seq[0] = f_seq[0]+f_seq[1];
    f_seq[1] = tmp-f_seq[1];

    if(length == 2) return;
    for(size_t i=0;i<(length/4-1);i++){
        size_t idx = i+1;
        T gr,gi,gnr,gni,s,c,tr0,ti0,tr1,ti1;
        std::tie(c,s) = omega_list[idx];
        LD_C(f_seq,idx,gr,gi);
        LD_C(f_seq,length/2-idx,gnr,gni);
        R2C_EPILOG(gr,gi,gnr,gni,s,c,tr0,ti0,tr1,ti1);
        ST_C(f_seq,idx,gr,gi);
        ST_C(f_seq,length/2-idx,gnr,gni);
    }
    if(length/4){
        T s,c;
        std::tie(c,s) = omega_list[length/4];
        f_seq[2*(length/4)] = f_seq[2*(length/4)] + f_seq[2*(length/4)+1]*c;
        f_seq[2*(length/4)+1] = -1*f_seq[2*(length/4)+1]*s;
    }
}

int main(){
#define FFT_LEN 32
#if 0
    float ts[2*FFT_LEN];
    float fs[2*FFT_LEN];
    //rand_vec(ts,2*FFT_LEN);
    for(size_t i=0;i<2*FFT_LEN;i++) ts[i] = i;

    fft_naive_mt(ts,fs,FFT_LEN);
    dump_vector(ts,2*FFT_LEN);
    dump_vector(fs,2*FFT_LEN);
    fft_cooley_tukey_r_mt(ts, FFT_LEN);
    dump_vector(ts,2*FFT_LEN);
    valid_vector(fs,ts,2*FFT_LEN);
#endif
    float ts[FFT_LEN];
    float ts2[2*FFT_LEN];
    float fs[FFT_LEN];
    float fs2[2*FFT_LEN];
    rand_vec(ts,FFT_LEN);
    fft_r2c_mt(ts,fs,FFT_LEN);
    {
        for(size_t i=0;i<FFT_LEN;i++){
            ts2[2*i] = ts[i];
            ts2[2*i+1] = 0;
        }
        fft_naive_mt(ts2,fs2,FFT_LEN);
        fs2[1] = fs2[FFT_LEN];
    }
    valid_vector(fs,fs2,FFT_LEN);
    //dump_vector(fs,FFT_LEN);
    //dump_vector(fs2,2*FFT_LEN);
}