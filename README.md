# FFT

fft, ifft, r2c, c2r, fft_2d_r2c, ifft_2d_c2r, convolve_1d/2d, correlation_1d/2d, tiling fft implementation.

using `build.sh` or `build.bat` in each sub directory to build on linux/windows

* [fft.cc](fft.cc)  
  naive fft/ifft, cooley_tukey/cooley_tukey_r, 1d/2d, r2c/c2r, convolve/correlate (signal process) implementation
* [mt/mt.cc](mt/mt.cc)  
  same as above, but not using [complex_t](complex.h) structure for complex arith. Plus implement AI/ML convolution(with r2c/c2r), notice the pad/shift/rotate is needed to go back to signal process convolution theorem.  
  This is more friendly for further gpu kernel impl.
* [fft_convolution_algo/fft_conv.h](fft_convolution_algo/fft_conv.h)  
  One header function for AI/ML convolution implementation using convolution theorem, support nchw/cnhw format. much of code copied from `mt/mt.cc`
* [fft_tiling/fft_tiling.cc](fft_tiling/fft_tiling.cc)  
  Tiling fft algorithm. Using small size fft to construct big size fft.

# r2c

r2c/c2r is algorithm to reduce compute complexity when all input is real. (real input time domain have hermitian symetry in freq domain)

```
suppose g(n) is N point real input, origin fourier transfor require pack image part to 0:
g(n) = x+0*j, 0...N-1

pack every continuous even/odd point to form N/2 complex x(n):
x(n) = xe+xo*j, 0...N/2-1

we want to compute:
G(k) = sigma[n]( (x+0*j)*(c+s*j)), c=cos(-2PI*k*n/N), s=sin(-2PI*k*n/N)
     = sigma[n]( x*c + x*s*j ), 0...0...N-1

Gr(k) = sigma[n]( x[n]*cos(-2PI*k*n/N))
Gi(k) = sigma[n]( x[n]*sin(-2PI*k*n/N)) *j, k,n=0...N-1

but we first compute x(n)->X(K), which only have half the complex number:
X(k) = sigma[n]( (xe+xo*j)*(cc+ss*j) ), cc=cos(-4PI*k*n/N), ss=sin(-4PI*k*n/N)
     = sigma[n]( xe*cc-xo*ss +  (xe*ss+xo*cc)*j ), 0...N/2-1

Xr(k) = sigma[n](x[2*n]*cos(-4PI*k*n/N)-x[2*n+1]*sin(-4PI*k*n/N))
Xi(K) = sigma[n](x[2*n]*sin(-4PI*k*n/N)+x[2*n+1]*cos(-4PI*k*n/N)) *j,  k,n=0...N/2-1


Note the symetric property:
     theta(k) = -4PI*k*n/N, theta(N/2-k) = -4PI*(N/2-k)*n/N = 4PIk*n/N - 2PI*n = 4PIk*n/N = -theta
     cc(k) = cc(N/2-k), ss(k) = -ss(N/2-k)

X(N/2-k) = sigma[n]( xe*cc+xo*ss +  (-xe*ss+xo*cc)*j )
Xr(N/2-k) = sigma[n](x[2*n]*cos(-4PI*k*n/N)+x[2*n+1]*sin(-4PI*k*n/N))
Xi(N/2-k) = sigma[n](-x[2*n]*sin(-4PI*k*n/N)+x[2*n+1]*cos(-4PI*k*n/N)) *j,  k,n=0...N/2-1

Then we need to figure out G(k) serial from X(k)
suppose, ar,ai,br,bi

Let:
G(k) = (ar+ai*j)*X(k) + (br+bi*j)*conj(X(N/2-k))
Gr(k) = ar*Xr(k)-ai*Xi(k) + br*Xr(N/2-k)+bi*Xi(N/2-k)
Gi(k) = ai*Xr(k)+ar*Xi(k) + bi*Xr(N/2-k)-br*Xi(N/2-k)

Here why use conj(X(N/2-k))? if X(N/2-k) the following formula will have no answer.

Gr(N/2-k) = x[n]*cos(-2PI*k*n/N) = 
x[2*n]*cos(-2PI*k*(2*n)/N)+x[2*n+1]*cos(-2PI*k*(2*n+1)/N) 
    = ar*(x[2*n]*cos(-4PI*k*n/N)-x[2*n+1]*sin(-4PI*k*n/N))
    - ai*(x[2*n]*sin(-4PI*k*n/N)+x[2*n+1]*cos(-4PI*k*n/N))
    + br*(x[2*n]*cos(-4PI*k*n/N)+x[2*n+1]*sin(-4PI*k*n/N))
    + bi*(-x[2*n]*sin(-4PI*k*n/N)+x[2*n+1]*cos(-4PI*k*n/N))

Gi(N/2-k) = x[n]*sin(-2PI*k*n/N) = 
x[2*n]*sin(-2PI*k*(2*n)/N)+x[2*n+1]*sin(-2PI*k*(2*n+1)/N) 
    = ai*(x[2*n]*cos(-4PI*k*n/N)-x[2*n+1]*sin(-4PI*k*n/N))
    + ar*(x[2*n]*sin(-4PI*k*n/N)+x[2*n+1]*cos(-4PI*k*n/N))
    + bi*(x[2*n]*cos(-4PI*k*n/N)+x[2*n+1]*sin(-4PI*k*n/N))
    - br*(-x[2*n]*sin(-4PI*k*n/N)+x[2*n+1]*cos(-4PI*k*n/N))

given sin(x+y) = sin(x)cos(y)+cos(x)sin(y), cos(x+y) = cos(x)cos(y)-sin(x)sin(y)
      sin(2x) = 2*sin(x)cos(x), cos(2x) = cos(x)^2 - sin(x)^2

let t0=-4PI*k*n/N, t1=-2PI*k/N, c0=cos(t0), s0=sin(t0, c1=cos(t1), s1=sin(t1)
    1) ar*c0-ai*s0+br*c0-bi*s0=c0
    2) -ar*s0-ai*c0+br*s0+bi*c0=c0*c1-s0*s1,  cos(t0+t1)
    3) ai*c0+ar*s0+bi*c0+br*s0=s0
    4) -ai*s0+ar*c0+bi*s0-br*c0=s0*c1+s1*c0,  sin(t0+t1)

solve above:
    4)-1), 2*bi*s0-2*br*c0 = s0*c1+s1*c0-c0
    3)+2), 2*bi*c0+2*br*s0 = c0*c1-s0*s1+s0
  ->
    br = 0.5*(1-s1) = 0.5*(1-sin(-2PI*k/N))
    bi = 0.5*c1 = 0.5*cos(-2PI*k/N)

    4)+1), 2*ar*c0-2*ai*s0 = s0*c1 + s1*c0 + c0
    3)-2), 2*ar*s0+2*ai*c0 = s0 - c0*c1 + s0*s1
  ->
    ar = 0.5*(1+s1) = 0.5*(1+sin(-2PI*k/N))
    ai = -0.5*c1 = -0.5*cos(-2PI*k/N)

  Note that, ar/ai/br/bi is independent of [n]. hence safe to plug into sigma[n]

Hence:
    Gr(k) = 0.5*((1+s)*Xr(k)+c*Xi(k) + (1-s)*Xr(N/2-k) + c*Xi(N/2-k))
    Gi(k) = 0.5*(-c*Xr(k)+(1+s)*Xi(k) + c*Xr(N/2-k)-(1-s)*Xi(N/2-k))
                    c=cos(-2PI*k/N), s=sin(-2PI*k/N)
                    for k=0...N/2-1, 
    and Xr(N/2)=Xr(0), Xi(N/2)=Xi(0)
when k==0: (s=0,c=1)
    Gr(0) = Xr(0)+Xi(0)
    Gi(0) = 0
when k=N/2: (s=0,c=-1)
    Gr(0) = Xr(0)-Xi(0)
    Gi(0) = 0
```
