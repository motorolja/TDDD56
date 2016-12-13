#!/bin/bash

 gcc hello_world_cl.c CLutilities.c -lOpenCL -I/usr/local/cuda/include -o hello_world_cl

