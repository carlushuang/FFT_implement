

import numpy as np
import scipy as sp
import scipy.signal
import sys
import math

def is_pow_of(n, p):
    # check value n is power of p
    x = n
    while x % p == 0:
        x /= p
    return x == 1

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

def cas_cs(n, k):
    theta = 2 * np.pi * k / n   # notice no minus sign here!
    return np.cos(theta), np.sin(theta)

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

def radix_2_fht(vec):
    length = len(vec)
    assert is_pow_of(length, 2), 'length must be power of 2'
    for j in range(length//4):
        c, s = cas_cs(length, j+1)
        # u = vec[length//2 + j + 1]
        # v = vec[length - j - 1]
        # u, v = c*u + s*v, s*u - c*v
        # vec[length//2 + j + 1] = u
        # vec[length - j - 1] = v
        tmp0 = s * vec[length//2 + j + 1]
        vec[length//2 + j + 1] *= c
        vec[length//2 + j + 1] += s * vec[length - j - 1]
        vec[length - j - 1] = -c * vec[length - j - 1] + tmp0
    for j in range(length//2):
        # u = vec[j]
        # v = vec[length//2 + j]
        # vec[j] = u + v
        # vec[length//2 + j] = u - v
        vec[j] =  vec[j] + vec[length//2 + j]
        vec[length//2 + j] = vec[j] - 2*vec[length//2 + j]

def radix_hft_select(r):
    if r == 2:
        return radix_2_fht

    assert False

def radix_index_reverse(length, r):
    '''
    https://ieeexplore.ieee.org/document/1165252
    index reverse of radix-r is:\
    1) n = r^k, there are n indexes
    2) for input index i=0...n-1, construct it into base of r, which have k digits
        i -> (dk-1, dk-2, ...,d0)(base r)
    3) reverse each digit, and get output index j:
        j -> (d0, d1, ...dk-1)(base r)
    4) get back to base-10 of j
    '''
    def base_10_to_base_r(base10_value, num_digits, base_r):
        # return list of digits, list size is num_digits, list[0] is lsb
        assert base_r ** num_digits > base10_value
        digits = [0] * num_digits
        v = base10_value
        i = 0
        while v:
            digits[i] = v % base_r
            v = v // base_r
            i += 1
        return digits
    def base_r_to_base_10(base_r_digits, num_digits, base_r):
        # return int, base_r_digits is list of size num_digits
        # contains digits of base_r. base_r_digits[0] is lsb
        value = 0
        for d in range(num_digits):
            value += (r**d) * base_r_digits[d]
        return value
    assert is_pow_of(length, r), 'length:{} must be power of {}'.format(length,r)
    reverse_list = [0] * length
    digits = int(math.log(length, r))
    for i in range(length):
        base_r_digits = base_10_to_base_r(i, digits, r)
        base_r_digits.reverse()
        j = base_r_to_base_10(base_r_digits, digits, r)
        reverse_list[i] = j
    return reverse_list

def radix_r_hft(vec, r):
    # for simplicity, pass in r be radix-r
    n = len(vec)
    assert is_pow_of(n, r), 'length:{} must be power of {}'.format(n,r)
    # r^d = n
    d = int(math.log(n, r))
    fvec = np.ndarray(n, dtype = vec.dtype)
    rindex = radix_index_reverse(n, r)
    # print('reverse index for length:{}, base:{}\n  ->{}'.format(n,r,rindex))
    for i in range(len(rindex)):
        fvec[i] = vec[rindex[i]]    # index reverse in front
    radix_caller = radix_hft_select(r)
    for i in range(d):
        li = r**(i+1)
        ki = n // li
        for j in range(ki):
            # print('slice:[{}:{}], j:{}'.format(j*li, (j+1)*li, ki))
            radix_caller(fvec[j*li : (j+1)*li])
    return fvec

if __name__ == '__main__':
    n = 16
    seq = np.random.random(n)
    #seq = np.random.random(n * 2).view(np.complex)
    fhseq = naive_dht_1d(seq)
    np.set_printoptions(precision=3)
    # print(seq)
    print(fhseq)
    print(radix_r_hft(seq, 2))
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