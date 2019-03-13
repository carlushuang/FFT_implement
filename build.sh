#!/bin/sh
SRC=fft.cc
TARGET=fft
CXX=g++
CXX_FLAGS="-O2 -Wall -lm -std=c++11 "

rm -rf $TARGET
$CXX $SRC $CXX_FLAGS -o $TARGET