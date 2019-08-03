#!/bin/sh
SRC=conv.cc
TARGET=conv.exe
CXX=g++
CXX_FLAGS="-O2 -Wall -std=c++11 "
#CXX_FLAGS="$CXX_FLAGS -g"
LDFLAGS="-lm "


rm -rf $TARGET
$CXX $SRC $CXX_FLAGS $LDFLAGS -o $TARGET
