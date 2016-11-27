// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

const int N = 16;
const int blocksize = 16;

__global__
void square(float *c)
{
  c[threadIdx.x] *= c[threadIdx.x];
}

int main()
{
  float*  host = (float*)malloc(N * sizeof(float));
  float* device;
  cudaMalloc(&device,  N * sizeof(float));

  // initialize the input data
  for (int i = 0; i<N; i++)
    {
      host[i] = (float)i;
    }

  cudaMemcpy(device, host, N*sizeof(float), cudaMemcpyHostToDevice);

  // do CPU calculations
  for (int i = 0; i<N; i++)
  {
    printf("CPU, index %d: %f\n",i,host[i]*host[i]);
  }
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );

  // do GPU calculations
  square <<< dimGrid, dimBlock >>> (device);
	cudaThreadSynchronize();
  cudaMemcpy(host, device, N*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i<N; i++)
    {
      printf("GPU, index %d: %f\n",i,host[i]);
    }

  free(host);
  cudaFree(device);

  return EXIT_SUCCESS;
}
