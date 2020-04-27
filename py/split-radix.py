import numpy as np
import scipy as sp
import scipy.signal
import sys

def omega(n, k):
    '''
    n is total pixel, k is current requested frequency at k.
    '''
    theta = -2 * np.pi * k / n
    return np.complex(np.cos(theta), np.sin(theta))

def omega_x(n, k, x):
    # omega at input x
    theta = -2 * np.pi * k * x / n
    return np.complex(np.cos(theta), np.sin(theta))

def fft1d_naive(vec):
    # assert type(vec) is list
    length = len(vec)
    fout = np.ndarray(length, dtype = np.complex)
    for k in range(length):
        fk = 0
        for t in range(length):
            fk += vec[t] * omega_x(length, k, t)
        fout[k] = fk
    return fout

def srfft1d_naive(vec):
    # http://www.fftw.org/~athena/papers/newsplit-pub.pdf
    assert len(vec) % 4 == 0
    length = len(vec)
    vec_2 = vec[0:length:2]
    vec_4p1 = vec[1:length:4]
    vec_4n1 = vec[3:(length-1):4]
    vec_4n1 = np.insert(vec_4n1, 0, vec[length-1])

    # print(vec)
    # print(vec_2)
    # print(vec_4p1)
    # print(vec_4n1)

    fvec_2 = fft1d_naive(vec_2)
    fvec_4p1 = fft1d_naive(vec_4p1)
    fvec_4n1 = fft1d_naive(vec_4n1)

    fvec = np.ndarray(length, dtype = np.complex)
    for k in range(length//4):
        fvec[k]             = fvec_2[k] + (omega(length, k) * fvec_4p1[k] + omega(length, -1*k) * fvec_4n1[k])
        fvec[k + length//2] = fvec_2[k] - (omega(length, k) * fvec_4p1[k] + omega(length, -1*k) * fvec_4n1[k])
        fvec[k + (length//4)*1] = fvec_2[k + N//4 ] - (1j)*(omega(length, k) * fvec_4p1[k] - omega(length, -1*k) * fvec_4n1[k])
        fvec[k + (length//4)*3] = fvec_2[k + N//4 ] + (1j)*(omega(length, k) * fvec_4p1[k] - omega(length, -1*k) * fvec_4n1[k])
    return fvec

if __name__ == '__main__':
    N = 8
    seq = np.random.random(N * 2).view(np.complex)
    print(seq)
    seq_f = fft1d_naive(seq)
    print(seq_f)
    print(np.fft.fft(seq))
    print('----')
    seq_srf = srfft1d_naive(seq)
    print(seq_srf)

