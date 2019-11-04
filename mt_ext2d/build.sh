#!/bin/sh
SRC=mt_ext2d.cc
TARGET=mt_ext2d.exe
CXX=g++
CXX_FLAGS="-O2 -Wall -std=c++11 "
CXX_FLAGS="$CXX_FLAGS -g"
LDFLAGS="-lm "

USE_FFTW=1
if [ "x$USE_FFTW" = "x1" ];then
    CXX_FLAGS="$CXX_FLAGS -DUSE_FFTW -I/usr/local/include"
    LDFLAGS="$LDFLAGS -L/usr/local/lib -Wl,-Bstatic -lfftw3 -Wl,-Bdynamic"
fi

rm -rf $TARGET
$CXX $SRC $CXX_FLAGS $LDFLAGS -o $TARGET
