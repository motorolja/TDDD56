#!/bin/bash

gcc hello_world_cl.c CLutilities.c -lOpenCL -I/usr/local/cuda/include -o hello_world_cl
gcc milli.c readppm.c CLutilities.c filter.c -lOpenCL -I/usr/local/cuda/include -lGL -lglut -o filter
