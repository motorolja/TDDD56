#!/bin/bash

gcc milli.c readppm.c CLutilities.c filter.c -L/usr/local/cuda/lib64 -lOpenCL -I/usr/local/cuda/include -lGL -lglut -o filter
