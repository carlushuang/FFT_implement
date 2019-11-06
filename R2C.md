# r2c/c2r 1d

1d r2c, input N real point, output N/2+1 complex point. If not use r2c, will result in N complex point.

```
[r2c]
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

Here why use conj(X(N/2-k))? X(N/2-k) is just the same...

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
                    c=cos(-2PI*k/N), s=sin(-2PI*k/N), k=0...N/2-1, 
    and Xr(N/2)=Xr(0), Xi(N/2)=Xi(0)
when k==0: (s=0,c=1)
    Gr(0) = Xr(0)+Xi(0)
    Gi(0) = 0
when k=N/2: (s=0,c=-1)
    Gr(0) = Xr(0)-Xi(0)
    Gi(0) = 0

[c2r]
  Xr(k) = 0.5*( Gr(k)*(1-sin) – Gi(k)*cos + Gr(N/2–k)*(1+sin) - Gi(N/2–k)*cos )
  Xi(k) = 0.5*( Gi(k)*(1-sin) + Gr(k)*cos - Gr(N/2–k)*cos – Gi(N/2–k)*(1+sin) )
```

# r2c/c2r ext2d
when do 2d real input fft/ifft, one can use (suppose N*N real input):
* first 1d horizontal r2c for each row, result in `N row, N/2+1 column` complex, then vertical `N-point` complex fft for each column, result in `N row, N/2+1 column` complex.
* pack every continue point as a complex, one can get `N row, N/2 column` complex, then do vertical `N-point` complex fft, then do horizontal `N/2-point` complex fft, result in `N row, N/2 column` complex. then use following fomular to get final `N row, N/2+1 column` complex.
```
[r2c_ext2d]
every row:
xe(n), xo(n)

merge every continuous item in a row, and view vertical:
gv(v) = xe(v)+xo(v)*j
Gv(vk) = sigma[v]( (xe(v)+xo(v)*j)*(cv+sv*j) )
       = sigma[v]( xe(v)*cv-xo(v)*sv + (xo(v)*cv+xe(v)*sv)*j)
        cv = cos(-2*PI*v*vk/N), sv = sin(-2*PI*v*vk/N)

Xe(vk) = sigma[v]( xe(v) * (cv+sv*j) )
       = sigma[v]( xe(v)*cv+xe(v)*sv*j)
Xo(vk) = sigma[v]( xo(v) * (cv+sv*j) )
       = sigma[v]( xo(v)*cv+xo(v)*sv*j)

we can get:
Gv(vk) = Xe(vk) + Xo(vk)*j
Gv(N-vk) = conj(Xe(vk)) + conj(Xo(vk))*j

so for every row:
Gh(n) = Xhe(n) +Xho(n)*j
Gh(n) = Xh(2*n) +Xh(2*n+1)*j = Xhr(2*n)+Xhi(2*n)*j+Xhr(2*n+1)*j-Xhi(2*n+1)
->
Ghr(n) = Xhr(2*n)-Xhi(2*n+1)
Ghi(n) = Xhr(2*n+1)+Xhi(2*n), n=0...N/2-1

# the N-vk row in current column
Gh_vk(N-vk) = conj(Xe(vk)) + conj(Xo(vk))*j = Xhr(2*n)-Xhi(2*n)*j+Xhr(2*n+1)*j+Xhi(2*n+1)
Ghr_vk(n) = Xhr(2*n)+Xhi(2*n+1)
Ghi_vk(n) = Xhr(2*n+1)-Xhi(2*n), n=0...N/2-1

for every row, let Xh(n) be fft after xe(n)+0*j, xo(n)+0*j, vertically.
the second horizontal phase should be:

XH(k) = sigma[n]( (Xhr(n)+Xhi(n)*j)*(c+s*j)), c=cos(-2*PI*n*k/N), s=sin(-2*PI*n*k/N), n=0...N-1

let ct=cos(-2*PI*k/N), st=sin(-2*PI*k/N)

XHr(k) = sigma[n]( Xhr(n)*c-Xhi(h)*s )  , n=0...N-1
       = sigma[n]( Xhr(2*n)*cos(-2*PI*2*n*k/N) + Xhr(2*n+1)*cos(-2*PI*(2*n+1)*k/N)
                  -Xhi(2*n)*sin(-2*PI*2*n*k/N) - Xhi(2*n+1)*sin(-2*PI*(2*n+1)*k/N)), n=0...N/2-1
       = sigma[n]( Xhr(2*n)*c2 + Xhr(2*n+1)*(c2*ct-s2*st) - Xhi(2*n)*s2 - Xhi(2*n+1)*(c2*st+s2*ct) )


XHi(k) = sigma[n]( Xhr(n)*s + Xhi(n)*c), n=0...N-1
       = sigma[n]( Xhr(2*n)*sin(-2*PI*2*n*k/N) + Xhr(2*n+1)*sin(-2*PI*(2*n+1)*k/N)
                  +Xhi(2*n)*cos(-2*PI*2*n*k/N) + Xhi(2*n+1)*cos(-2*PI*(2*n+1)*k/N)), n=0...N/2-1
       = sigma[n]( Xhr(2*n)*s2 + Xhr(2*n+1)*(c2*st+s2*ct) + Xhi(2*n)*c2 + Xhi(2*n+1)*(c2*ct-s2*st) )

GH(k) = sigma[n]( (Xh(2*n) +Xh(2*n+1)*j)*(c2+s2*j)), c2=cos(-4*PI*n*k/N), s2=sin(-4*PI*n*k/N), n=0...N/2-1
      = sigma[n]( ( Xhr(2*n)-Xhi(2*n+1))*c2 - (Xhr(2*n+1)+Xhi(2*n))*s2
                + ((( Xhr(2*n)-Xhi(2*n+1))*s2 + (Xhr(2*n+1)+Xhi(2*n))*c2))*j )

GHr(k) = sigma[n]( (Xhr(2*n)-Xhi(2*n+1))*c2 - (Xhr(2*n+1)+Xhi(2*n))*s2), n=0...N/2-1
GHi(k) = sigma[n]( (Xhr(2*n)-Xhi(2*n+1))*s2 + (Xhr(2*n+1)+Xhi(2*n))*c2), n=0...N/2-1

GH_vk(k) = sigma[n]( (Xhr(2*n)+Xhi(2*n+1) + (Xhr(2*n+1)-Xhi(2*n))*j)*(c2+s2*j)),
GHr_vk(k) = sigma[n]( (Xhr(2*n)+Xhi(2*n+1))*c2 - (Xhr(2*n+1)-Xhi(2*n))*s2), n=0...N/2-1
GHi_vk(k) = sigma[n]( (Xhr(2*n)+Xhi(2*n+1))*s2 + (Xhr(2*n+1)-Xhi(2*n))*c2), n=0...N/2-1

# the N-vk row in N/2-k column
GHr_vkhk(N/2-k) = sigma[n]( (Xhr(2*n)+Xhi(2*n+1))*c2 + (Xhr(2*n+1)-Xhi(2*n))*s2), n=0...N/2-1
GHi_vkhk(N/2-k) = sigma[n]( -(Xhr(2*n)+Xhi(2*n+1))*s2 + (Xhr(2*n+1)-Xhi(2*n))*c2), n=0...N/2-1

XH(k) = (wr+wi*j)*GH(k) + (zr+zi*j)*conj(GH_vkhk(k))
XHr(k)+XHi(k) = (wr+wi*j)*(GHr+GHi*j) + (zr+zi*j)*(GHr_vkhk-GHi_vkhk*j)

XHr(k) = wr*GHr - wi*GHi + zr*GHr_vkhk + zi*GHi_vkhk
XHi(k) = wr*GHi + wi*GHr - zr*GHi_vkhk + zi*GHr_vkhk

Xhr(2*n)*cos(-2*PI*2*n*k/N) + Xhr(2*n+1)*cos(-2*PI*(2*n+1)*k/N)
    -Xhi(2*n)*sin(-2*PI*2*n*k/N) - Xhi(2*n+1)*sin(-2*PI*(2*n+1)*k/N)
    =wr*((Xhr(2*n)-Xhi(2*n+1))*c2 - (Xhr(2*n+1)+Xhi(2*n))*s2)
    -wi*((Xhr(2*n)-Xhi(2*n+1))*s2 + (Xhr(2*n+1)+Xhi(2*n))*c2)
    +zr*((Xhr(2*n)+Xhi(2*n+1))*c2 + (Xhr(2*n+1)-Xhi(2*n))*s2)
    +zi*(-(Xhr(2*n)+Xhi(2*n+1))*s2 + (Xhr(2*n+1)-Xhi(2*n))*c2)

    1)  wr*c2 - wi*s2 + zr*c2 - zi*s2 = c2
    2) -wr*s2 - wi*c2 + zr*s2 + zi*c2 = c2*ct - s2*st
    3) -wr*s2 - wi*c2 - zr*s2 - zi*c2 = -s2
    4) -wr*c2 + wi*s2 + zr*c2 - zi*s2 = -(c2*st + ct*s2)

-> {wr: st/2 + 1/2, wi: -ct/2, zr: -st/2 + 1/2, zi: ct/2}

Xhr(2*n)*sin(-2*PI*2*n*k/N) + Xhr(2*n+1)*sin(-2*PI*(2*n+1)*k/N)
    +Xhi(2*n)*cos(-2*PI*2*n*k/N) + Xhi(2*n+1)*cos(-2*PI*(2*n+1)*k/N)
    =wr*((Xhr(2*n)-Xhi(2*n+1))*s2 + (Xhr(2*n+1)+Xhi(2*n))*c2)
    +wi*((Xhr(2*n)-Xhi(2*n+1))*c2 - (Xhr(2*n+1)+Xhi(2*n))*s2)
    -zr*(-(Xhr(2*n)+Xhi(2*n+1))*s2 + (Xhr(2*n+1)-Xhi(2*n))*c2)
    +zi*((Xhr(2*n)+Xhi(2*n+1))*c2 + (Xhr(2*n+1)-Xhi(2*n))*s2)

    1)  wr*s2 + wi*c2 + zr*s2 + zi*c2 = s2
    2)  wr*c2 - wi*s2 - zr*c2 + zi*s2 = c2*st + ct*s2
    3)  wr*c2 - wi*s2 + zr*c2 - zi*s2 = c2
    4) -wr*s2 - wi*c2 + zr*s2 + zi*c2 = c2*ct - s2*st

-> {wr: st/2 + 1/2, wi: -ct/2, zr: -st/2 + 1/2, zi: ct/2}

Hence:
    XHr(k) = 0.5*((1+st)*GHr + ct*GHi + (1-st)*GHr_vkhk + ct*GHi_vkhk)
    XHi(k) = 0.5*((1+st)*GHi - ct*GHr - (1-st)*GHi_vkhk + ct*GHr_vkhk)
                            ct=cos(-2*PI*k/N), st=sin(-2*PI*k/N)
1)
for column 0 , GH_vkhk = GH_vk, hence:
    XHr(0) = 0.5*((1+st)*GHr + ct*GHi + (1-st)*GHr_vk + ct*GHi_vk)
           = 0.5*(GHr + GHi + GHr_vk + GHi_vk)
    XHi(0) = 0.5*((1+st)*GHi - ct*GHr - (1-st)*GHi_vk + ct*GHr_vk)
           = 0.5*(GHi - GHr - GHi_vk + GHr_vk)
2)
for column N/2:
    XHr(N/2) = 0.5*(GHr - GHi + GHr_vk - GHi_vk)
    XHi(N/2) = 0.5*(GHi + GHr - GHi_vk - GHr_vk)
3)
for column k row 0,  GH = GH_vk, hence:
    XHr(k) = 0.5*((1+st)*GHr + ct*GHi + (1-st)*GHr_hk + ct*GHi_hk)
    XHi(k) = 0.5*((1+st)*GHi - ct*GHr - (1-st)*GHi_hk + ct*GHr_hk)
4)
for column 0 row 0:
    XHr(N/2) = GHr + GHi
    XHi(N/2) = 0
5)
for column N/2 row 0:
    XHr(N/2) = GHr - GHi
    XHi(N/2) = 0


[c2r_ext2d]

GHr(k) = 0.5*((1+st)*XHr - ct*XHi + (1-st)*XHr_vkhk - ct*XHi_vkhk)
GHi(k) = 0.5*((1+st)*XHi + ct*XHr - (1-st)*XHi_vkhk - ct*XHr_vkhk)
                        ct=cos(-2*PI*k/N), st=sin(-2*PI*k/N)

1) for row 0:
    GHr(k) = 0.5*((1+st)*XHr - ct*XHi + (1-st)*XHr_hk - ct*XHi_hk)
    GHi(k) = 0.5*((1+st)*XHi + ct*XHr - (1-st)*XHi_hk - ct*XHr_hk)

2) other:
    GHr(k) = 0.5*((1+st)*XHr - ct*XHi + (1-st)*XHr_vkhk - ct*XHi_vkhk)
    GHi(k) = 0.5*((1+st)*XHi + ct*XHr - (1-st)*XHi_vkhk - ct*XHr_vkhk)

```
