// Matrix addition, GPU version

#include <stdio.h>

__global__
void add_matrix(float *a, float *b, float *c, int N)
{
	int index;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
      {
        index = i + j*N;
        c[index] = a[index] + b[index];
      }
}

__global__
void add_matrix_tid(float *a, float *b, float *c, int N)
{
	int x, y, index;

  // global index for threads
  x = blockIdx.x * blockDim.x + threadIdx.x;
  y = blockIdx.y * blockDim.y + threadIdx.y;

  index = x + y*N;
  c[index] = a[index] + b[index];
  /*
    printf("gridDim.x = %f, gridDim.y = %f, threadIdx.x = %f, threadIdx.y = %f, blockDim.x = %f, blockIdx.x = %f\n",
    gridDim.x, gridDim.y, threadIdx.x, threadIdx.y, blockDim.x, blockIdx.x);
  */
}

int main()
{
	const int N = 32;

  float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];
  float *c_d, *a_d, *b_d;
  cudaMalloc(&a_d,  N * N * sizeof(float));
  cudaMalloc(&b_d,  N * N * sizeof(float));
  cudaMalloc(&c_d,  N * N * sizeof(float));

  // get information about the hardware
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("Device name: %s\n", prop.name);
    printf("Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("MaxThreadsBlock: %d\n", prop.maxThreadsPerBlock);
    printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
    printf("TotalGlobalMem / sizeof(float): %d\n", prop.totalGlobalMem/sizeof(float));
    printf("MaxGridSize: %d\n", prop.maxGridSize);
    printf("MaxThreadDim:(%d,%d,%d) \n", (prop.maxThreadsDim[0]),
           (prop.maxThreadsDim[1]),(prop.maxThreadsDim[2]));
  }

  // initialize matrix a & b
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        {
          a[i+j*N] = 10 + i;
          b[i+j*N] = (float)j / N;
        }
    }

  // create events and set starting point for timer
  cudaEvent_t myEvent,myEventB;
  cudaEventCreate(&myEvent);
  cudaEventCreate(&myEventB);
  // copy matrixes to GPU
  cudaMemcpy(a_d, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, N*N*sizeof(float), cudaMemcpyHostToDevice);

  int block_size = 64, grid_size = block_size/N;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(grid_size,grid_size);
  // do GPU calculations
  cudaEventRecord(myEvent, 0);
  cudaEventSynchronize(myEvent);

  add_matrix_tid <<< dimGrid, dimBlock>>> (a_d, b_d, c_d, N);
  cudaThreadSynchronize();

  // set end point for timer and get the elapsed time
  cudaEventRecord(myEventB, 0);
  cudaEventSynchronize(myEventB);
  float theTime;
  cudaEventElapsedTime(&theTime, myEvent, myEventB);

  // Overwrite a with the result
  cudaMemcpy(c, c_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  /*
    for (int i = 0; i < N; i++)
    {
		for (int j = 0; j < N; j++)
		{
    printf("%0.2f ", c[i+j*N]);
		}
		printf("\n##\n");
    }
  */
  printf("time in ms: %f \n",theTime);
  delete(a);
  delete(b);
  delete(c);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
}
