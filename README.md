# FFT
fft/ifft, r2c/c2r, 2d_r2c/2d_c2r, convolve, correlation, tiling fft, srfft, pfa, radix-2/3/5

using `build.sh` or `build.bat` in each sub directory to build on linux/windows

* [fft.cc](fft.cc)  
  naive fft/ifft, cooley_tukey/cooley_tukey_r, 1d/2d, r2c/c2r, convolve/correlate (signal process) implementation
* [mt/mt.cc](mt/mt.cc)  
  same as above, but not using [complex_t](complex.h) structure for complex arith. Plus implement AI/ML convolution(with r2c/c2r), notice the pad/shift/rotate is needed to go back to signal process convolution theorem.  
  This is more friendly for further gpu kernel impl.
* [mt_ext2d/mt_ext2d.cc](mt_ext2d/mt_ext2d.cc)  
  special optimization for 2d real input fft/ifft. the compute complexity is the same as original 1d r2c/c2r, but this make 2d r2c/c2r easier to write.
* [fft_convolution_algo/fft_conv.h](fft_convolution_algo/fft_conv.h)  
  One header function for AI/ML convolution implementation using convolution theorem, support nchw/cnhw format. much of code copied from `mt/mt.cc`
* [fft_tiling/fft_tiling.cc](fft_tiling/fft_tiling.cc)  
  Tiling fft algorithm. Using small size fft to construct big size fft.
* [py/split-radix.py](py/split-radix.py)  
  srfft
* [py/radix-r.py](py/radix-r.py)  
  implement radix-2, radix-3, radix-5
* [py/pfa.py](py/pfa.py)  
  prime-factor-algorithm
* [py/dht.py](py/dht.py)  
  hartley transform

# r2c
[r2c/c2r](http://processors.wiki.ti.com/index.php/Efficient_FFT_Computation_of_Real_Input) is a algorithm for all real input fft,  which can reduce half compute complexity. Check [R2C.md](R2C.md) for detail.

