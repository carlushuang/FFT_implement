# PFA https://enacademic.com/dic.nsf/enwiki/151599
# https://pdfs.semanticscholar.org/18e9/67f8f17ef30e6ab8f77b0f6fe56b0af4abd4.pdf
# Prime-factor algorithm
import numpy as np
import scipy as sp
import scipy.signal
import sys

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

def gcd(p, q):
    x, y = p, q
    while(y):
        x, y = y, x % y
    return x

def linear_congruence(a, b, m):
    # http://gauss.math.luc.edu/greicius/Math201/Fall2012/Lectures/linear-congruences.article.pdf
    # solve congruence equation for:
    # a*x ~= b (mod m)
    if b == 0:
        return 0
    assert a>0 and b>0 and m>0
    d = gcd(a, m)
    if b % d != 0:
        print('have no solution for {}*x ~= {}(mod {})'.format(a,b,m))
        return 0        
    for x in range(m):
        if (a*x - b) % m == 0:
            break
    if d == 1:
        return x
    # if d not 1, there are multiple solutions, and has form x0 + (m/d)t
    # return a list
    x_list = []
    for t in range(d):
        x_list.append(x + (m//d) * t)
    return x_list

def pfa_fft1d_naive_n1_n2(vec, n1, n2):
    def solve_mapping_congruence_diophantine(map_n1, map_n2):
        # http://volta.sdsu.edu/~amir/Bhagat18High.pdf
        # need solve equation to find m1, m2
        # n1*m1 + n2*m2 ~= 1 mod (n1*n2)
        #   -> n1*m1 ~= 1 mod n2
        #   -> n2*m2 ~= 1 mod n1
        map_m1 = linear_congruence(map_n1, 1, map_n2)
        map_m2 = linear_congruence(map_n2, 1, map_n1)
        # print('[dio] n1:{}, n2:{}, m1:{}, m2:{}'.format(map_n1,map_n2,map_m1,map_m2))
        assert map_m1 != 0 and map_m2 != 0
        if type(map_m1) is list:
            map_m1 = map_m1[0]
        if type(map_m2) is list:
            map_m2 = map_m2[0]
        
        assert (map_n1 * map_m1 + map_n2 * map_m2) % (map_n1 * map_n2) == 1
        return map_m1, map_m2

    # ruritanian correspondence mapping
    def rcm_fwd_mapping(map_n1, map_n2):
        # from n1, n2 to n
        assert gcd(map_n1, map_n2) == 1, "need be co-prime"
        mapping = np.ndarray((map_n1, map_n2), dtype=np.int)
        for in1 in range(map_n1):
            for in2 in range(map_n2):
                mapping[in1][in2] = (in1 * map_n2 + in2 * map_n1) % (map_n1 * map_n2)
        return mapping
    def rcm_bwd_mapping(map_n1, map_n2):
        # from n to n1, n2
        assert gcd(map_n1, map_n2) == 1, "need be co-prime"
        map_m1, map_m2 = solve_mapping_congruence_diophantine(map_n1, map_n2)
        mapping = np.ndarray((map_n1*map_n2, 2), dtype=np.int)
        for n in range(map_n1 * map_n2):
            in1 = (n * map_m2) % map_n1
            in2 = (n * map_m1) % map_n2
            mapping[n][0] = in1
            mapping[n][1] = in2
        return mapping

    #  Chinese remainder theorem
    def crt_fwd_mapping(map_n1, map_n2):
        '''
        from n1, n2 to n
        '''
        assert gcd(map_n1, map_n2) == 1, "need be co-prime"
        mapping = np.ndarray((map_n1, map_n2), dtype=np.int)
        map_m1, map_m2 = solve_mapping_congruence_diophantine(map_n1, map_n2)
        for in1 in range(map_n1):
            for in2 in range(map_n2):
                mapping[in1][in2] = (in1 * map_n2 * map_m2 + in2 * map_n1 * map_m1) % (map_n1 * map_n2)
        return mapping

    def crt_bwd_mapping(map_n1, map_n2):
        '''
        from n to n1, n2
        '''
        assert gcd(map_n1, map_n2) == 1, "need be co-prime"
        mapping = np.ndarray((map_n1*map_n2, 2), dtype=np.int)
        for n in range(map_n1 * map_n2):
            in1 = n % map_n1
            in2 = n % map_n2
            mapping[n][0] = in1
            mapping[n][1] = in2
        return mapping

    def map_1d_to_2d(vec_1d, mapping):
        '''
        note: it is easier if use fomular instead of an mapping array to do map.
              here is only for demostration
        '''
        map_n1 = mapping.shape[0]
        map_n2 = mapping.shape[1]
        assert len(vec_1d) == (map_n1 * map_n2)
        vec_2d = np.ndarray((map_n1, map_n2), dtype = vec_1d.dtype)
        for in1 in range(map_n1):
            for in2 in range(map_n2):
                vec_2d[in1][in2] = vec_1d[mapping[in1][in2]]
        return vec_2d
    assert len(vec) == n1 * n2
    assert gcd(n1, n2) == 1, "need be co-prime"

    def map_2d_to_1d(vec_2d, mapping):
        map_n1 = vec_2d.shape[0]
        map_n2 = vec_2d.shape[1]
        assert len(mapping) == (map_n1 * map_n2)
        vec_1d = np.ndarray(map_n1 * map_n2, dtype = vec_2d.dtype)
        for n in range(map_n1 * map_n2):
            in1 = mapping[n][0]
            in2 = mapping[n][1]
            vec_1d[n] = vec_2d[in1][in2]
        return vec_1d

    if 0:
        input_mapping  = crt_fwd_mapping(n1, n2)
        output_mapping = rcm_bwd_mapping(n1, n2)
    else:
        # this stratagy is better, for no need to solve congruence diophantine
        # if this is performed offline, then they just the same
        input_mapping  = rcm_fwd_mapping(n1, n2)
        output_mapping = crt_bwd_mapping(n1, n2)
    #print('input mapping')
    #print(input_mapping)
    #print('output mapping')
    #print(output_mapping)

    seq_2d = map_1d_to_2d(vec, input_mapping)

    fseq_2d = np.ndarray((n1, n2), dtype = vec.dtype)
    for k in range(n1):
        fseq_2d[k] = np.fft.fft(seq_2d[k])

    fseq_2d = fseq_2d.transpose()
    for i in range(n2):
        fseq_2d[i] = np.fft.fft(fseq_2d[i])
    fseq_2d = fseq_2d.transpose()

    return map_2d_to_1d(fseq_2d, output_mapping)

if __name__ == '__main__':
    N = 100
    N1, N2 = 4, 25
    
    seq = np.random.random(N * 2).view(np.complex)
    fseq = pfa_fft1d_naive_n1_n2(seq, N1, N2)
    np.set_printoptions(precision=3)
    fseq_ref = np.fft.fft(seq)
    #print(np.fft.fft(seq))
    #print(fseq)
    print(valid_array(fseq_ref, fseq))

    