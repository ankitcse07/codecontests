#!/bin/csh -f

\rm -f output
g++ codecontest.cxx -march=native -O3 -fopenmp -mtune=generic -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -msse4 -std=c++0x

