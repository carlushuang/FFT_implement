# http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.50.8814
# some small radix-r implementations
import numpy as np
import scipy as sp
import scipy.signal
import sys
import math

def valid_array(lhs,rhs,atol=0.00001):
    r = np.allclose(lhs,rhs,atol)
    return r

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

def is_pow_of(n, p):
    # check value n is power of p
    x = n
    while x % p == 0:
        x /= p
    return x == 1

def radix_5(vec):
    # inplace and, need tmp register(complex)
    length = len(vec)
    assert is_pow_of(length, 5), 'length must be power of 5'
    c1 = 1/4
    c2 = np.sqrt(5)/4
    c3 = np.sqrt( (5-np.sqrt(5)) / (5+np.sqrt(5)) )
    c4 = (1/2) * np.sqrt(5/2 + np.sqrt(5)/2)
    for j in range(length // 5):
        z0 = vec[j]
        z1 = omega(length, j) * vec[j+length//5]
        z2 = omega(length, 2*j) * vec[j+(length*2)//5]
        s1 = z1 - omega(length, 4*j) * vec[j+(length*4)//5]
        s2 = 2*z1 - s1
        s3 = z2 - omega(length, 3*j) * vec[j+(length*3)//5]
        s4 = 2*z2 - s3
        s5 = s2 + s4
        s6 = s2 - s4
        s7 = z0 - c1*s5
        s8 = s7 - c2*s6
        s9 = 2*s7 - s8
        s10 = s1 + c3*s3
        s11 = c3*s1 - s3
        vec[j] = z0 + s5
        # t1 = s9 - 1j * c4 * s10
        # vec[j + length//5] = 2*s9 - t1
        # t2 = s8 - 1j*c4*s11
        # vec[j + (length*2)//5] = 2*s8 - t2
        # vec[j + (length*3)//5] = t2
        # vec[j + (length*4)//5] = t1
        t1 = s9 - 1j * c4 * s10
        vec[j + (length*4)//5] = 2*s9 - t1
        t2 = s8 - 1j*c4*s11
        vec[j + (length*3)//5] = 2*s8 - t2
        vec[j + (length*2)//5] = t2
        vec[j + (length*1)//5] = t1

def radix_3(vec):
    # inplace and, 1 extra tmp register(complex)
    length = len(vec)
    assert is_pow_of(length, 3), 'length must be power of 3'
    c1 = -1/2
    c2 = (np.sqrt(3)/2) * (-1j)
    for j in range(length // 3):
        '''
        z1 = omega(length, j) * vec[j + length//3]
        s1 = z1 - omega(length, 2*j)*vec[j + (length*2)//3]
        s2 = 2*z1 - s1
        s3 = s2 + vec[j]
        s4 = vec[j] + c1*s2
        s5 = s4 - c2*s1
        s6 = 2*s4 - s5
        vec[j] = s3
        vec[j+length//3] = s6
        vec[j+(2*length)//3] = s5
        '''
        vec[j + length//3]      = omega(length, j) * vec[j + length//3]
        vec[j + (length*2)//3]  = vec[j + length//3] - omega(length, 2*j)*vec[j + (length*2)//3]
        vec[j + length//3]      = 2 * vec[j + length//3]  - vec[j + (length*2)//3]
        tmp0                    = vec[j] + c1 * vec[j + length//3]
        vec[j]                  = vec[j + length//3] + vec[j]
        vec[j + (length*2)//3]  = tmp0 - c2 * vec[j + (length*2)//3]
        vec[j + length//3]      = 2*tmp0 - vec[j + (length*2)//3]

def radix_2(vec):
    # inplace, no temp is needed
    length = len(vec)
    assert is_pow_of(length, 2), 'length must be power of 2'
    for j in range(length // 2):
        # print(' radix_2 for {}, {}-{}'.format(length,j,j+length//2))
        vec[j+length//2] = vec[j] - omega(length, j) * vec[j + length//2]
        vec[j] = 2*vec[j] -  vec[j+length//2]

def radix_select(r):
    if r == 2:
        return radix_2
    if r == 3:
        return radix_3
    if r == 5:
        return radix_5
    assert False

def gen_radix_reverse_index(length, r):
    assert is_pow_of(length, r), 'length:{} must be power of {}'.format(length,r)
    index = []
    for k in range(length // r):
        for q in range(r):
            index.append(k+q*(length//2))
    return index

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

def radix_r(vec, r):
    # for simplicity, pass in r be radix-r
    n = len(vec)
    assert is_pow_of(n, r), 'length:{} must be power of {}'.format(n,r)
    # r^d = n
    d = int(math.log(n, r))
    fvec = np.ndarray(n, dtype = vec.dtype)
    rindex = radix_index_reverse(n, r)
    print('reverse index for length:{}, base:{}\n  ->{}'.format(n,r,rindex))
    for i in range(len(rindex)):
        fvec[i] = vec[rindex[i]]    # index reverse in front
    radix_caller = radix_select(r)
    for i in range(d):
        li = r**(i+1)
        ki = n // li
        for j in range(ki):
            # print('slice:[{}:{}], j:{}'.format(j*li, (j+1)*li, ki))
            radix_caller(fvec[j*li : (j+1)*li])
    return fvec

if __name__ == '__main__':
    r = 5
    p = 2
    n = r**p
    seq = np.random.random(n * 2).view(np.complex)
    fseq = radix_r(seq, r)
    fseq_ref = np.fft.fft(seq)
    np.set_printoptions(precision=3)
    print(fseq)
    print(fseq_ref)
    print(valid_array(fseq_ref, fseq))
    