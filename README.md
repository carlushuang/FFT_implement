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