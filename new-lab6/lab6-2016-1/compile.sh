#!/bin/sh
gcc milli.c readppm.c CLutilities.c filter.c -lOpenCL -I/usr/local/cuda/include -lGL -lglut -o filter
