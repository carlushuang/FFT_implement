::used in windows, conda install -c msys2 m2w64-toolchain
@echo off
set SRC=fft_tiling.cc
set TARGET=fft_tiling
set CXX=g++
set CXX_FLAGS=-O2 -Wall -lm -std=c++11
::set CXX_FLAGS=%CXX_FLAGS% -g

if exist %TARGET% del %TARGET%
%CXX% %SRC% %CXX_FLAGS% -o %TARGET%
