import numpy as np
import scipy as sp
import scipy.signal

def valid_array(lhs,rhs,atol=0.0001):
    r = np.allclose(lhs,rhs,atol)
    return r

def v_fourier_linear():
    '''
    linear: F(a1xb1 + a2xb2) = F(a1)*F(b1) + F(a2)*F(b2)
        x   : convolution
        *   : elementwise product
        F() : fourier transform
    '''
    len_data=8
    len_filter=3
    len_out=len_data+len_filter-1

    def linear_1d():
        d1=np.random.randint(0,10,size=len_data)
        f1=np.random.randint(0,10,size=len_filter)
        d2=np.random.randint(0,10,size=len_data)
        f2=np.random.randint(0,10,size=len_filter)

        o1=np.convolve(d1,f1,'full')
        o2=np.convolve(d2,f2,'full')
        o_lhs = o1+o2

        d1p = np.pad(d1,(0,len_out-len_data),'constant')
        d2p = np.pad(d2,(0,len_out-len_data),'constant')
        f1p = np.pad(f1,(0,len_out-len_filter),'constant')
        f2p = np.pad(f2,(0,len_out-len_filter),'constant')

        o1_ = np.fft.ifft(np.fft.fft(d1p)*np.fft.fft(f1p)).real
        o2_ = np.fft.ifft(np.fft.fft(d2p)*np.fft.fft(f2p)).real
        o_rhs = o1_+o2_

        valid = valid_array(o_lhs, o_rhs)
        print("v_fourier_linear 1d:{}".format(valid))

    def linear_2d():
        d1=np.random.randint(0,10,size=(len_data,len_data))
        f1=np.random.randint(0,10,size=(len_filter,len_filter))
        d2=np.random.randint(0,10,size=(len_data,len_data))
        f2=np.random.randint(0,10,size=(len_filter,len_filter))

        o1=sp.signal.convolve2d(d1,f1,'full')
        o2=sp.signal.convolve2d(d2,f2,'full')
        o_lhs = o1+o2

        d1p = np.pad(d1,((0,len_out-len_data),(0,len_out-len_data)),'constant')
        d2p = np.pad(d2,((0,len_out-len_data),(0,len_out-len_data)),'constant')
        f1p = np.pad(f1,((0,len_out-len_filter),(0,len_out-len_filter)),'constant')
        f2p = np.pad(f2,((0,len_out-len_filter),(0,len_out-len_filter)),'constant')

        o1_ = np.fft.ifft2(np.fft.fft2(d1p)*np.fft.fft2(f1p)).real
        o2_ = np.fft.ifft2(np.fft.fft2(d2p)*np.fft.fft2(f2p)).real
        o_rhs = o1_+o2_

        valid = valid_array(o_lhs, o_rhs)
        print("v_fourier_linear 2d:{}".format(valid))


    linear_1d()
    linear_2d()

    
if __name__ == '__main__':
    v_fourier_linear()