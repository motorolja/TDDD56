#!/bin/bash
nvcc -lglut -lGL mandel.cu -o mandel-gpu

exit
