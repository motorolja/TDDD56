#!/bin/bash

nvcc bitonic_main.cpp bitonic_kernel.cu milli.c -o bitonic_gpu
