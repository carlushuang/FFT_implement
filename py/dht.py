

import numpy as np
import scipy as sp
import scipy.signal
import sys
import math

def valid_array(lhs,rhs,atol=0.00001):
    r = np.allclose(lhs,rhs,atol)
    return r

def cas(n, k):
    '''
    n is total pixel, k is current requested frequency at k.
    '''
    theta = 2 * np.pi * k / n   # notice no minus sign here!
    return np.cos(theta) + np.sin(theta)

def cas_x(n, k, x):
    '''
    cas at input x
    '''
    theta = 2 * np.pi * k * x / n   # notice no minus sign here!
    return np.cos(theta) + np.sin(theta)

def naive_dht_1d(vec):
    # this is unnormalized version of hartley
    # input vec can be real only, and complex
    length = len(vec)
    fvec = np.ndarray(length, dtype = vec.dtype)
    for k in range(length):
        fk = 0
        for t in range(length):
            fk += vec[t] * cas_x(length, k, t)
        fvec[k] = fk
    return fvec

def dht_to_dft(vec, r2c_half_mode = False):
    length = len(vec)
    convert_length = (length // 2 + 1) if r2c_half_mode else length
    dft_vec = np.ndarray(convert_length, dtype = np.complex)    # force to complex
    # print('dht_to_dft, len:{}, cvt:{}'.format(length, convert_length))
    if vec.dtype == np.complex:
        # complex
        for i in range(convert_length):
            ir = (length - i) % length
            dr = 0.5*(vec[i].real + vec[ir].real + vec[i].imag - vec[ir].imag)
            di = 0.5*(vec[i].imag + vec[ir].imag - vec[i].real + vec[ir].real)
            dft_vec[i] = complex(dr, di)
    else:
        # real only
        for i in range(convert_length):
            ir = (length - i) % length
            dr = 0.5*(vec[i] + vec[ir])
            di = -0.5*(vec[i] - vec[ir])
            dft_vec[i] = complex(dr, di)
    return dft_vec

def r2c_1d_with_naive_dht(vec):
    assert vec.dtype != np.complex
    fhseq = naive_dht_1d(seq)
    return dht_to_dft(fhseq, True)

if __name__ == '__main__':
    n = 22
    seq = np.random.random(n)
    #seq = np.random.random(n * 2).view(np.complex)
    fhseq = naive_dht_1d(seq)
    np.set_printoptions(precision=3)
    print(seq)
    print(fhseq)
    #print(naive_dht_1d(fhseq) / n)
    print('-------------')
    fseq_ref = np.fft.fft(seq)
    fseq_h   = dht_to_dft(fhseq)
    print(fseq_ref)
    print(fseq_h)
    print(valid_array(fseq_ref, fseq_h))

    r2c_fseq = r2c_1d_with_naive_dht(seq)
    # print(r2c_fseq)
    print(valid_array(fseq_ref[0:(len(fseq_ref)//2)+1], r2c_fseq))