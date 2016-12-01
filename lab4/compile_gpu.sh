#!/bin/bash
nvcc mandel.cu -o mandel-gpu -lglut -lGL

exit
