#ifndef __FFT_CONV_H
#define __FFT_CONV_H

/*
* n: batch
* c: input channel
* h: input height
* w: input width
* k: filter count
* r: filter_h
* s: filter_w
* p: pad_h
* q: pad_w
* u: stride_h
* v: stride_w
* l: dilation_h
* j: dilation_w
*
*/

/* use config in mt:
#define PRE_PAD_DATA
#define FFTCONV_USE_CONJ // this is a good mode that all omega use the same function, unified_omega_func_f32
#define FFTCONV_USE_CONJ_NO_ROTATE // this mode, all kernel padding shape is same. we restore output in c2r part
//#define FFTCONV_USE_CONJ_A  // same as FFTCONV_USE_CONJ, but notice, time reverse is fft shift
#define MERGE_2D_NYQUEST_FREQ
*/

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

template<typename T>
static inline void fft_r2c_mt(const T* t_seq, T * f_seq, size_t length, bool merge_nyquist_freq=false){
    if(length == 1) return;
    assert( ((length & (length - 1)) == 0 ) && "current only length power of 2");
    T tmp;
    auto omega_func = [](size_t total_n, size_t k){
        T theta = C_2PI*k / total_n;
        return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta)); };

    std::vector<std::tuple<T,T>> omega_list;
    omega_list.resize(length/2);
    for(size_t i=0;i<length/2;i++) omega_list[i] = omega_func(length,i);

    std::vector<size_t> brev;
    bit_reverse_permute(log2(length/2), brev);
    for(size_t i=0;i<length;i++) f_seq[i] = t_seq[i];
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
template<typename T>
void ifft_c2r_mt(T* t_seq, const T * f_seq, size_t length, bool merge_nyquist_freq=false){
    // the 0-th and length/2-th complex number, image part must be zero, same as fftw
    if(length == 1) return;
    assert( ((length & (length - 1)) == 0 ) && "current only length power of 2");

    auto omega_func = [](size_t total_n, size_t k){
        T theta = -1*C_2PI*k / total_n;
        return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta));
    };

    std::vector<std::tuple<T,T>> omega_list;
    omega_list.resize(length/2);
    for(size_t i=0;i<length/2;i++){
        omega_list[i] = omega_func(length,i);
    }
    if(length == 2) return;

    if(!merge_nyquist_freq){
        t_seq[0] = 0.5*(f_seq[0]+f_seq[length]);
        t_seq[1] = 0.5*(f_seq[0]-f_seq[length]);
    }else{
        t_seq[0] = 0.5*(f_seq[0]+f_seq[1]);
        t_seq[1] = 0.5*(f_seq[0]-f_seq[1]);
    }

    for(size_t i=1;i<=(length/4-1);i++){
        T xr,xi,xnr,xni,s,c,sr0,si0,sr1,si1;
        std::tie(c,s) = omega_list[i];

        LD_C(f_seq,i,xr,xi);
        LD_C(f_seq,length/2-i,xnr,xni);
        IC2R_EPILOG(xr,xi,xnr,xni,s,c,sr0,si0,sr1,si1);
        ST_C(t_seq,i,xr,xi);
        ST_C(t_seq,length/2-i,xnr,xni);
    }
    if(length/4){
        t_seq[2*(length/4)] = f_seq[2*(length/4)];
        t_seq[2*(length/4)+1] = -1*f_seq[2*(length/4)+1];
    }
    ifft_cooley_tukey_r_mt(t_seq, length/2, true);
}
template<typename T>
void ifft2d_c2r_mt(T* t_seq, const T * f_seq, size_t seq_w, size_t seq_h){
    bool h_merge_nyquist_freq = true;
    size_t v_len = h_merge_nyquist_freq?seq_h:(seq_h+2);
    T * seq = new T[v_len*seq_w];
    float * f_seq_first_row = NULL;
    if(h_merge_nyquist_freq){
        f_seq_first_row = new float[2*seq_w];
        for(size_t w=0;w<seq_w;w++){
            f_seq_first_row[2*w] = f_seq[2*w]-f_seq[(seq_h/2)*2*seq_w+2*w+1 ];
            f_seq_first_row[2*w+1] = f_seq[2*w+1]+f_seq[(seq_h/2)*2*seq_w+2*w ];
        }
    }

    // horizontal
    T * h_even = new T[seq_w];
    T * h_odd  = new T[seq_w];
    auto omega_func = [](size_t total_n, size_t k){
        T theta = -1*C_2PI*k / total_n;
        return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta));
    };

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
            seq[h*2*seq_w+2*w] = (h_even[2*w]+h_odd[2*w]*c+h_odd[2*w+1]*s)/2;
            seq[h*2*seq_w+2*w+1] = (h_even[2*w+1]-h_odd[2*w]*s+h_odd[2*w+1]*c)/2;
            seq[h*2*seq_w+seq_w+2*w] = (h_even[2*w]-h_odd[2*w]*c-h_odd[2*w+1]*s)/2;
            seq[h*2*seq_w+seq_w+2*w+1] = (h_even[2*w+1]+h_odd[2*w]*s-h_odd[2*w+1]*c)/2;

        }
    }
    delete [] h_even;
    delete [] h_odd;

    if(h_merge_nyquist_freq) delete [] f_seq_first_row;

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
template<typename T>
static inline void fft2d_r2c_mt(const T* t_seq, T * f_seq, size_t seq_w, size_t seq_h){
    bool h_merge_nyquist_freq=true;
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
            f_seq[h*2*seq_w+2*w] = h_even[2*w]+h_odd[2*w]*c-h_odd[2*w+1]*s;
            f_seq[h*2*seq_w+2*w+1] = h_even[2*w+1]+h_odd[2*w]*s+h_odd[2*w+1]*c;

            f_seq[h*2*seq_w+seq_w+2*w] = h_even[2*w]-h_odd[2*w]*c+h_odd[2*w+1]*s;
            f_seq[h*2*seq_w+seq_w+2*w+1] = h_even[2*w+1]-h_odd[2*w]*s-h_odd[2*w+1]*c;
        }
    }
    if(h_merge_nyquist_freq){
        f_seq[0] = f_seq[0];
        f_seq[(seq_h/2)*2*seq_w] = f_seq[1];
        f_seq[1] = 0;
        f_seq[(seq_h/2)*2*seq_w+1] = 0;

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
}
static inline void fft_conv_fwd_nchw(const float *src, const float *filter, float *dst,
    size_t n, size_t c, size_t h, size_t w, size_t k, size_t r, size_t s, size_t p, size_t q, size_t u, size_t v, size_t l, size_t j)
{
    size_t in,ik,ic;
    size_t oh = fft_conv_out_size(h, p, l, r, u);
    size_t ow = fft_conv_out_size(w, q, j, s, v);
    size_t seq_pad_h = (size_t)std::pow(2, std::ceil(std::log2(h + r-1)));
    size_t seq_pad_w = (size_t)std::pow(2, std::ceil(std::log2(w + s-1)));
    size_t fft_h = seq_pad_h/2+1;
    size_t fft_w = 2*seq_pad_w;

    assert((r-1)>=p); assert((s-1)>=q);
    assert(u==1 && v==1 && l==1 && j==1 && "currently only support unit stride/dilation");

    float * seq_data    = new float[seq_pad_h*seq_pad_w];
    float * seq_filter  = new float[seq_pad_h*seq_pad_w];
    float * fft_data    = new float[fft_h*fft_w];
    float * fft_filter  = new float[fft_h*fft_w];
    float * fft_out     = new float[fft_h*fft_w];
    float * dst_pad     = new float[seq_pad_h*seq_pad_w];

    for(size_t ii=0;ii<seq_pad_h*seq_pad_w;ii++){ seq_data[ii] = 0; seq_filter[ii] = 0; }

    for(in=0;in<n;in++){
        for(ik=0;ik<k;ik++){
            // clear oh*ow
            for(size_t ii=0;ii<oh*ow;ii++) dst[in*k*oh*ow+ik*oh*ow+ii]=0;
            for(ic=0;ic<c;ic++){
                for(size_t jj=0;jj<r;jj++)
                    for(size_t ii=0;ii<s;ii++)
                        seq_filter[jj*seq_pad_w+ii] = filter[ik*c*r*s+ic*r*s+jj*s+ii];
                for(size_t jj=0;jj<h;jj++)
                    for(size_t ii=0;ii<w;ii++)
                        seq_data[(jj+p)*seq_pad_w+ii+q] = src[in*c*h*w+ic*h*w+jj*w+ii];

                // 1: fft data, fft filter
                fft2d_r2c_mt(seq_data, fft_data, seq_pad_w, seq_pad_h);
                fft2d_r2c_mt(seq_filter, fft_filter, seq_pad_w, seq_pad_h);

                // 2: element wise multiply
                for(size_t jj=0;jj<fft_h;jj++){
                    for(size_t ii=0;ii<fft_w/2;ii++){
                    fft_out[jj*fft_w+2*ii]   = fft_data[jj*fft_w+2*ii]*fft_filter[jj*fft_w+2*ii] + fft_data[jj*fft_w+2*ii+1]*fft_filter[jj*fft_w+2*ii+1];
                    fft_out[jj*fft_w+2*ii+1] = -1*fft_data[jj*fft_w+2*ii]*fft_filter[jj*fft_w+2*ii+1] + fft_data[jj*fft_w+2*ii+1]*fft_filter[jj*fft_w+2*ii];
                    }
                }

                // 3: ifft output
                ifft2d_c2r_mt(dst_pad, fft_out, seq_pad_w, seq_pad_h);

                for(size_t jj=0;jj<oh;jj++){
                    for(size_t ii=0;ii<ow;ii++){
                        dst[in*k*oh*ow+ik*oh*ow+jj*ow+ii] += dst_pad[jj*seq_pad_w+ii];
                    }
                }
            }
        }
    }

    delete []  seq_data;
    delete []  seq_filter;
    delete []  fft_data;
    delete []  fft_filter;
    delete []  fft_out;
    delete []  dst_pad;
}

#endif