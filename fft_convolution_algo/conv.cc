#include <stdio.h>
#include <time.h>
//#include "fft_conv.h"
#include "fft_conv_ext2d.h"
#include "naive_conv.h"
static int64_t out_size(int64_t in_size, int64_t pad, int64_t dilation, int64_t ksize, int64_t stride)
{
     return (in_size + 2*pad- dilation*(ksize-1) -1)/stride + 1;
}
void rand_vector(float * vec, size_t num){
    static size_t inited=0;
    if(!inited){ inited = 1; srand (time(NULL));}
    for(size_t i=0;i<num;i++) vec[i] = ((float)(rand()%1000))/1000.0f;
}
size_t valid_vector(float *lhs, float *rhs, size_t num, float delta=0.02){
    size_t err_cnt=0;
#define ABS(x)  ((x>0)?x:(-1*x))
    for(size_t i=0;i<num;i++){
        float d = lhs[i] - rhs[i];
        d = ABS(d);
        if(d>delta) {printf("diff at %3lu, lhs:%f, rhs:%f, diff:%f\n",i,lhs[i],rhs[i],d);err_cnt++;}
    }
    return err_cnt;
}
void dump_vector_nchw(float * t, size_t n, size_t c, size_t h, size_t w){
    size_t in,ic,ih,iw;
    for(in=0;in<n;in++){
        for(ic=0;ic<c;ic++){
            for(ih=0;ih<h;ih++){
                for(iw=0;iw<w;iw++){
                    printf("%.3f ",t[in*c*h*w+ic*h*w+ih*w+iw]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("--------------------------------\n");
    }
}
void dump_vector_cnhw(float * t, size_t n, size_t c, size_t h, size_t w){
    size_t in,ic,ih,iw;
    for(ic=0;ic<c;ic++){
        for(in=0;in<n;in++){
            for(ih=0;ih<h;ih++){
                for(iw=0;iw<w;iw++){
                    printf("%.3f ",t[ic*n*h*w+in*h*w+ih*w+iw]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("--------------------------------\n");
    }
}

size_t next_config(size_t *n, size_t *c, size_t *h, size_t *w, size_t *k, size_t *r, size_t *s, size_t *p, size_t *q, size_t *u, size_t *v, size_t *l, size_t *j){
    size_t n_arr[] ={1,2,4};
    size_t c_arr[] ={3,8,28,36};
    size_t wh_arr[]={7,25,77,128};
    size_t k_arr[] ={4,16,28};
    //size_t rs_arr[]={1,3,5,7,11};
    //size_t pq_arr[]={0,1,2,3};
    size_t r_arr[]={1,3,5,7};
    size_t s_arr[]={1,3,5,7};
    size_t p_arr[]={0,1,2,3};
    size_t q_arr[]={0,1,2,3};
    size_t uv_arr[]={1};
    size_t d_arr[] ={1};
    
    static size_t have_next=1;
    static size_t in=0;
    static size_t ic=0;
    static size_t iwh=0;
    static size_t ik=0;
    static size_t ir=0;
    static size_t is=0;
    static size_t ip=0;
    static size_t iq=0;
    static size_t iuv=0;
    static size_t id=0;
    size_t need_restart = 0;

    if(!have_next)
        return 0;

restart:
    if( out_size((int64_t)wh_arr[iwh],(int64_t) p_arr[ip], (int64_t)d_arr[id], (int64_t)r_arr[ir],(int64_t)uv_arr[iuv])<=0
        || out_size((int64_t)wh_arr[iwh],(int64_t) q_arr[iq], (int64_t)d_arr[id], (int64_t)s_arr[is],(int64_t)uv_arr[iuv])<=0
        || r_arr[ir]>wh_arr[iwh]
        || s_arr[is]>wh_arr[iwh]
        || (r_arr[ir]-1)<p_arr[ip]
        || (s_arr[is]-1)<q_arr[iq]
        ){
        need_restart = 1;
        goto next_cfg;
    }
    need_restart= 0;
    *n=n_arr[in];
    *c=c_arr[ic];
    *h=wh_arr[iwh];
    *w=wh_arr[iwh];
    *k=k_arr[ik];
    *r=r_arr[ir];
    *s=s_arr[is];
    *p=p_arr[ip];
    *q=q_arr[iq];
    *u=uv_arr[iuv];
    *v=uv_arr[iuv];
    *l=d_arr[id];
    *j=d_arr[id];
#if 0
    *n=4;
    *c=128;
    *h=17;
    *w=17;
    *k=128;
    *r=1;
    *s=7;
    *p=0;
    *q=3;
    *u=1;
    *v=1;
    *l=1;
    *j=1;
    have_next=0;
#endif
#define ARR_LEN(arr)  (sizeof(arr)/sizeof(arr[0]))
#define ITR_ELEM(elem)  i##elem++; if (i##elem >=ARR_LEN(elem##_arr) ){ i##elem=0;
next_cfg:
    ITR_ELEM(d)
        ITR_ELEM(uv)
            ITR_ELEM(p)
                ITR_ELEM(q)
                    ITR_ELEM(r)
                        ITR_ELEM(s)
                            ITR_ELEM(k)
                                ITR_ELEM(wh)
                                    ITR_ELEM(c)
                                        ITR_ELEM(n)
                                            have_next=0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if(need_restart)
        goto restart;
    return 1;
}

int main(){
    size_t n,c,h,w,k,r,s,p,q,u,v,l,j,oh,ow;
    
    //printf("n   c   h   w   k   r   s   p   q   u   v   l   j   oh  ow\n");
    while(next_config(&n, &c, &h, &w, &k, &r, &s, &p, &q, &u, &v, &l, &j)){
        size_t err_cnt;
        oh = out_size(h, p, l, r, u);
        ow = out_size(w, q, j, s, v);
        //printf("%-3lu %-3lu %-3lu %-3lu %-3lu %-3lu %-3lu %-3lu %-3lu %-3lu %-3lu %-3lu %-3lu %-3lu %-3lu ",
        //    n,c,h,w,k,r,s,p,q,u,v,l,j,oh,ow);
        printf("n:%-3lu c:%-3lu h:%-3lu w:%-3lu k:%-3lu r:%-3lu s:%-3lu p:%-3lu q:%-3lu u:%-3lu v:%-3lu l:%-3lu j:%-3lu oh:%-3lu ow:%-3lu, ",
            n,c,h,w,k,r,s,p,q,u,v,l,j,oh,ow);
            
        float * t_input = new float[n*c*h*w];
        float * t_out = new float[n*k*oh*ow];
        float * t_filter = new float[k*c*r*s];
        
        float * t_ref = new float[n*k*oh*ow];
        rand_vector(t_input, n*c*h*w);
        rand_vector(t_filter, k*c*r*s);
        fft_conv_fwd_nchw(t_input, t_filter, t_out, n,c,h,w,k,r,s,p,q,u,v,l,j);
        naive_conv_fwd_nchw(t_input, t_filter, t_ref, n,c,h,w,k,r,s,p,q,u,v,l,j);
        err_cnt = valid_vector(t_out, t_ref, n*k*oh*ow);
        printf("fwd:%s ",(err_cnt==0)?"y":"n");
        assert(err_cnt==0 && "fail to validate fwd");
        delete [] t_ref;
#if 0
        t_ref = new float[n*c*h*w];
        rand_vector(t_out, n*k*oh*ow);
        rand_vector(t_filter, k*c*r*s);
        mkldnn_conv_bwd_d_cnhw(t_input, t_filter, t_out, n,c,h,w,k,r,s,p,q,u,v,l,j);
        naive_conv_bwd_d_cnhw(t_ref, t_filter, t_out, n,c,h,w,k,r,s,p,q,u,v,l,j);
        err_cnt = valid_vector(t_input, t_ref, n*c*h*w);
        printf("%s ",(err_cnt==0)?"y":"n");
        assert(err_cnt==0 && "fail to validate bwd_d");
        delete [] t_ref;

        t_ref = new float[k*c*r*s];
        rand_vector(t_input, n*c*h*w);
        rand_vector(t_out, n*k*oh*ow);
        mkldnn_conv_bwd_f_cnhw(t_input, t_filter, t_out, n,c,h,w,k,r,s,p,q,u,v,l,j);
        naive_conv_bwd_f_cnhw(t_input, t_ref, t_out, n,c,h,w,k,r,s,p,q,u,v,l,j);
        err_cnt = valid_vector(t_filter, t_ref, k*c*r*s, 0.05);
        printf("%s ",(err_cnt==0)?"y":"n");
        assert(err_cnt==0 && "fail to validate bwd_f");
        delete [] t_ref;
#endif   
        delete [] t_input;
        delete [] t_filter;
        delete [] t_out;
        printf("\n");
    }
    return 0;
}