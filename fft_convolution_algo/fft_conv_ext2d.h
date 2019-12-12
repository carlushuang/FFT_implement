#ifndef __FFT_CONV_EXT2D_H
#define __FFT_CONV_EXT2D_H

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
    gr = 0.5*(tr0 + ti0*s + tr1*c); gi = 0.5*(ti1 + tr1*s - ti0*c); \
    gnr = 0.5*(tr0 - ti0*s - tr1*c); gni = -0.5*(ti1 - tr1*s + ti0*c); }while(0)
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

template<typename T>
std::tuple<T,T> omega_func (size_t total_n, size_t k){
    T theta = -1*C_2PI*k / total_n;
    return std::make_tuple<T,T>((T)cos(theta), (T)sin(theta));
}

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

template<typename T>
static inline void _fft_cooley_tukey_r_mt(T * seq, size_t c_length, bool is_inverse_fft, bool need_final_reverse = true){
    if(c_length == 1) return;
    assert( ( (c_length & (c_length - 1)) == 0 ) && "current only length power of 2");

    for(size_t itr = 2; itr<=c_length; itr<<=1){
        size_t stride = c_length/itr;
        size_t groups = itr/2;
        size_t group_len = stride*2;
        std::vector<std::tuple<T,T>> omega_list; omega_list.resize(itr/2);
        for(size_t i = 0; i < itr/2 ; i ++) omega_list[i] = omega_func<T>( itr, i);
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
static inline void fft_r2c_ext2d_postproc_vh(const T* t_seq, T * f_seq, size_t length){
    // 2d fft, used in first vertical fft, then horizontal fft
    // t_seq is length in height, length in width float (length/2 complex)
    // f_seq will be length in height, length+2 in width float (length/2+1 complex)

    // column 0, row 0
    f_seq[0] = t_seq[0]+t_seq[1];
    f_seq[1] = (T)0;

    // column 0, row 1...length-1
    for(size_t r=1;r<=(length-1);r++){
        f_seq[r*(length+2)] = 0.5*(t_seq[r*length] + t_seq[r*length+1]
                                + t_seq[(length-r)*length] + t_seq[(length-r)*length+1] );
        f_seq[r*(length+2)+1] = 0.5*(t_seq[r*length+1] - t_seq[r*length]
                                - t_seq[(length-r)*length+1] + t_seq[(length-r)*length]);
    }

    // column 1...length/2-1, row 0
    //  XHr(k) = 0.5*((1+st)*GHr + ct*GHi + (1-st)*GHr_hk + ct*GHi_hk)
    //  XHi(k) = 0.5*((1+st)*GHi - ct*GHr - (1-st)*GHi_hk + ct*GHr_hk)
    for(size_t c=1;c<=(length/2-1);c++){
        T ct, st;
        std::tie(ct,st) = omega_func<T>(length, c);
        f_seq[2*c] = 0.5*((1+st)*t_seq[2*c] + ct*t_seq[2*c+1] 
                + (1-st)*t_seq[2*(length/2-c)] + ct*t_seq[2*(length/2-c)+1]);
        f_seq[2*c+1] = 0.5*((1+st)*t_seq[2*c+1] - ct*t_seq[2*c] 
                - (1-st)*t_seq[2*(length/2-c)+1] + ct*t_seq[2*(length/2-c)]);
    }

    // column 1...length/2-1, other row
    //   XHr(k) = 0.5*((1+st)*GHr + ct*GHi + (1-st)*GHr_vkhk + ct*GHi_vkhk)
    //   XHi(k) = 0.5*((1+st)*GHi - ct*GHr - (1-st)*GHi_vkhk + ct*GHr_vkhk)
    for(size_t r=1;r<=length-1;r++){
        for(size_t c=1;c<=(length/2-1);c++){
            T ct, st;
            std::tie(ct,st) = omega_func<T>(length, c);
            f_seq[r*(length+2)+c*2] = 0.5*((1+st)*t_seq[r*length+c*2] + ct*t_seq[r*length+c*2+1]
                            + (1-st)*t_seq[(length-r)*length+(length/2-c)*2]
                            + ct*t_seq[(length-r)*length+(length/2-c)*2+1]);
            f_seq[r*(length+2)+c*2+1] = 0.5*((1+st)*t_seq[r*length+c*2+1] - ct*t_seq[r*length+c*2]
                            - (1-st)*t_seq[(length-r)*length+(length/2-c)*2+1]
                            + ct*t_seq[(length-r)*length+(length/2-c)*2]);
        }
    }

    // column N/2, row 0
    f_seq[length] = t_seq[0] - t_seq[1];
    f_seq[length+1] = (T)0;

    // column N/2, other row
    //  XHr(N/2) = 0.5*(GHr - GHi + GHr_vk - GHi_vk)
    //  XHi(N/2) = 0.5*(GHi + GHr - GHi_vk - GHr_vk)
    for(size_t r=1;r<=length-1;r++){
        f_seq[r*(length+2) + length] = 0.5*( t_seq[r*length] - t_seq[r*length+1] 
                        + t_seq[(length-r)*length] - t_seq[(length-r)*length+1]);
        f_seq[r*(length+2) + length + 1] = 0.5*(t_seq[r*length+1] + t_seq[r*length]
                        - t_seq[(length-r)*length+1] - t_seq[(length-r)*length]);
    }
}
template<typename T>
static inline void ifft_c2r_ext2d_preproc_vh(T* t_seq, const T * f_seq, size_t length){
    // f_seq is length in height, length+2 in width float (length/2+1 complex)
    // t_seq will be length in height, length in width float (length/2 complex)

    // row 0, column 0...N/2-1
    // GHr(k) = 0.5*((1+st)*XHr - ct*XHi + (1-st)*XHr_hk - ct*XHi_hk)
    // GHi(k) = 0.5*((1+st)*XHi + ct*XHr - (1-st)*XHi_hk - ct*XHr_hk)
    for(size_t c=0;c<(length/2);c++){
        T ct, st;
        std::tie(ct,st) = omega_func<T>(length, c);
        t_seq[2*c] = 0.5*((1+st)*f_seq[2*c] - ct*f_seq[2*c+1]
                + (1-st)*f_seq[2*(length/2-c)] - ct*f_seq[2*(length/2-c)+1]);
        t_seq[2*c+1] = 0.5*((1+st)*f_seq[2*c+1] + ct*f_seq[2*c] +
                - (1-st)*f_seq[2*(length/2-c)+1] - ct*f_seq[2*(length/2-c)]);
    }

    // other row, column 0...N/2-1
    // GHr(k) = 0.5*((1+st)*XHr - ct*XHi + (1-st)*XHr_vkhk - ct*XHi_vkhk)
    // GHi(k) = 0.5*((1+st)*XHi + ct*XHr - (1-st)*XHi_vkhk - ct*XHr_vkhk)
    for(size_t r=1;r<=(length-1);r++){
        for(size_t c=0;c<(length/2);c++){
            T ct, st;
            std::tie(ct,st) = omega_func<T>(length, c);
            t_seq[r*length+2*c] = 0.5*((1+st)*f_seq[r*(length+2)+2*c] - ct*f_seq[r*(length+2)+2*c+1]
                    + (1-st)*f_seq[(length-r)*(length+2)+2*(length/2-c)] - ct*f_seq[(length-r)*(length+2)+2*(length/2-c)+1]);
            t_seq[r*length+2*c+1] = 0.5*((1+st)*f_seq[r*(length+2)+2*c+1] + ct*f_seq[r*(length+2)+2*c] +
                    - (1-st)*f_seq[(length-r)*(length+2)+2*(length/2-c)+1] - ct*f_seq[(length-r)*(length+2)+2*(length/2-c)]);
        }
    }
}

template<typename T>
static inline void fft2d_r2c_ext2d(const T* t_seq, T * f_seq, size_t seq_w, size_t seq_h){
    assert(seq_w == seq_h && "current only support w==h");
    T * t_seq2 = new T[seq_w*seq_h];
    // vertical
    T * vt = new T[seq_h*2];
    for(size_t w=0;w<(seq_w/2);w++){
        for(size_t h=0;h<seq_h;h++){
            vt[2*h+0] = t_seq[h*seq_w+2*w+0];
            vt[2*h+1] = t_seq[h*seq_w+2*w+1];
        }
        fft_cooley_tukey_r_mt(vt, seq_h);
        for(size_t h=0;h<seq_h;h++){
            t_seq2[h*seq_w+2*w+0] = vt[2*h+0];
            t_seq2[h*seq_w+2*w+1] = vt[2*h+1];
        }
    }
    delete [] vt;

    // horizontal
    for(size_t h=0;h<seq_h;h++)
        fft_cooley_tukey_r_mt(t_seq2+h*seq_w, seq_w/2);

    // postproc
    fft_r2c_ext2d_postproc_vh(t_seq2, f_seq, seq_h);

    delete [] t_seq2;
}
template<typename T>
static inline void ifft2d_c2r_ext2d(T* t_seq, const T * f_seq, size_t seq_w, size_t seq_h){
    assert(seq_w == seq_h && "current only support w==h");

    // preproc
    ifft_c2r_ext2d_preproc_vh(t_seq, f_seq, seq_w);

    // horizontal
    for(size_t h=0;h<seq_h;h++)
        ifft_cooley_tukey_r_mt(t_seq+h*seq_w, seq_w/2);

    // vertical
    T * vt = new T[seq_h*2];
    for(size_t w=0;w<(seq_w/2);w++){
        for(size_t h=0;h<seq_h;h++){
            vt[2*h+0] = t_seq[h*seq_w+2*w+0];
            vt[2*h+1] = t_seq[h*seq_w+2*w+1];
        }
        ifft_cooley_tukey_r_mt(vt, seq_h);
        for(size_t h=0;h<seq_h;h++){
            t_seq[h*seq_w+2*w+0] = vt[2*h+0];
            t_seq[h*seq_w+2*w+1] = vt[2*h+1];
        }
    }

    delete [] vt;
}

/**************************************************************************************/


static inline int64_t fft_conv_out_size(int64_t in_size, int64_t pad, int64_t dilation, int64_t ksize, int64_t stride)
{
     return (in_size + 2*pad- dilation*(ksize-1) -1)/stride + 1;
}


static inline void fft_conv_fwd_nchw(const float *src, const float *filter, float *dst,
    size_t n, size_t c, size_t h, size_t w, size_t k, size_t r, size_t s, size_t p, size_t q, size_t u, size_t v, size_t l, size_t j)
{
    size_t in,ik,ic;
    size_t oh = fft_conv_out_size(h, p, l, r, u);
    size_t ow = fft_conv_out_size(w, q, j, s, v);
    size_t seq_pad_h = (size_t)pow(2, ceil(log2(h + r-1)));
    size_t seq_pad_w = (size_t)pow(2, ceil(log2(w + s-1)));

    size_t fft_size = seq_pad_h>seq_pad_w?seq_pad_h:seq_pad_w;

    // vertial->horizontal
    size_t fft_h = fft_size;
    size_t fft_w = 2*(fft_size/2+1);

    assert((r-1)>=p); assert((s-1)>=q);
    assert(u==1 && v==1 && l==1 && j==1 && "currently only support unit stride/dilation");

    float * seq_data    = new float[fft_size*fft_size];
    float * seq_filter  = new float[fft_size*fft_size];
    float * fft_data    = new float[fft_h*fft_w];
    float * fft_filter  = new float[fft_h*fft_w];
    float * fft_out     = new float[fft_h*fft_w];
    float * dst_pad     = new float[fft_size*fft_size];

    for(size_t ii=0;ii<fft_size*fft_size;ii++){ seq_data[ii] = 0; seq_filter[ii] = 0; }

    for(in=0;in<n;in++){
        for(ik=0;ik<k;ik++){
            // clear oh*ow
            for(size_t ii=0;ii<oh*ow;ii++) dst[in*k*oh*ow+ik*oh*ow+ii]=0;
            for(ic=0;ic<c;ic++){
                for(size_t jj=0;jj<r;jj++)
                    for(size_t ii=0;ii<s;ii++)
                        seq_filter[jj*fft_size+ii] = filter[ik*c*r*s+ic*r*s+jj*s+ii];
                for(size_t jj=0;jj<h;jj++)
                    for(size_t ii=0;ii<w;ii++)
                        seq_data[(jj+p)*fft_size+ii+q] = src[in*c*h*w+ic*h*w+jj*w+ii];

                // 1: fft data, fft filter
                fft2d_r2c_ext2d(seq_data, fft_data, fft_size, fft_size);
                fft2d_r2c_ext2d(seq_filter, fft_filter, fft_size, fft_size);

                // 2: element wise multiply
                for(size_t jj=0;jj<fft_h;jj++){
                    for(size_t ii=0;ii<fft_w/2;ii++){
                    fft_out[jj*fft_w+2*ii]   = fft_data[jj*fft_w+2*ii]*fft_filter[jj*fft_w+2*ii] + fft_data[jj*fft_w+2*ii+1]*fft_filter[jj*fft_w+2*ii+1];
                    fft_out[jj*fft_w+2*ii+1] = -1*fft_data[jj*fft_w+2*ii]*fft_filter[jj*fft_w+2*ii+1] + fft_data[jj*fft_w+2*ii+1]*fft_filter[jj*fft_w+2*ii];
                    }
                }

                // 3: ifft output
                ifft2d_c2r_ext2d(dst_pad, fft_out, fft_size, fft_size);

                for(size_t jj=0;jj<oh;jj++){
                    for(size_t ii=0;ii<ow;ii++){
                        dst[in*k*oh*ow+ik*oh*ow+jj*ow+ii] += dst_pad[jj*fft_size+ii];
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