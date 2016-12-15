// Reduction lab, find maximum

#include <stdio.h>
#include "milli.c"

//#define SIZE 100000
#define SIZE 200000000
//#define SIZE 16

__global__ void find_max(int *data, int N)
{

  int idx;
  idx = threadIdx.x + blockDim.x*blockIdx.x;
	// Write your CUDA kernel here
  // max = first int in our range
  int local_max = data[idx];
  for (int i = 0; i < N; i++)
    {
      if (data[idx + i] > local_max)
        local_max = data[idx + i];
    }
  data[idx] = local_max;

  __syncthreads();
  // master thread synch all local max data in the block and the get biggest
  // This way we may get it later

  if (threadIdx.x == 0)
    {
      for (int i = 0; i < blockDim.x; i++)
        {
          if (data[idx + i * N] > local_max)
            local_max = data[idx + i * N];
        }
      data[idx] = local_max;
    }
}

// TODO: needs to consider outliers to be accurate..
void set_max_block(int* data, int grid_size, int block_size, int N)
{
  // init to first found max block value
  int local_max = data[0];
  for (int i = 0; i < grid_size; i++)
    {
      if (data[i * block_size * N] > local_max)
        local_max = data[i * block_size * N];
    }
  data[0] = local_max;
}


void launch_cuda_kernel(int *data, int N)
{
	// Handle your CUDA kernel launches in this function

	int *devdata;
	int size = sizeof(int) * N;
	cudaMalloc( (void**)&devdata, size);
	cudaMemcpy(devdata, data, size, cudaMemcpyHostToDevice );

  int block_size = 16, grid_size = 8;
  // Dummy launch
  dim3 dimBlock( block_size, 1 );
  dim3 dimGrid( grid_size, 1 );
  int nr_of_ints_thread = SIZE / grid_size / block_size;
  find_max<<<dimGrid, dimBlock>>>(devdata, nr_of_ints_thread);
  cudaThreadSynchronize();
  cudaError_t err = cudaPeekAtLastError();
  if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	// Only the result needs copying!
	cudaMemcpy(data, devdata, sizeof(int), cudaMemcpyDeviceToHost );
	cudaFree(devdata);
  //  set_max_block(data, grid_size, block_size, nr_of_ints_thread);
}

// CPU max finder (sequential)
void find_max_cpu(int *data, int N)
{
  int i, m;

	m = data[0];
	for (i=0;i<N;i++) // Loop over data
	{
		if (data[i] > m)
			m = data[i];
	}
	data[0] = m;
}

// Dummy data in comments below for testing
int data[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
int data2[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};

int main()
{
  // Generate 2 copies of random data
  srand(time(NULL));
  for (long i=0;i<SIZE;i++)
  {
    data[i] = rand() % (SIZE * 5);
    data2[i] = data[i];
  }

  // The GPU will not easily beat the CPU here!
  // Reduction needs optimizing or it will be slow.
  ResetMilli();
  find_max_cpu(data, SIZE);
  printf("CPU time %f\n", GetSeconds());
  ResetMilli();
  launch_cuda_kernel(data2, SIZE);
  printf("GPU time %f\n", GetSeconds());

  // Print result
  printf("\n");
  printf("CPU found max %d\n", data[0]);
  printf("GPU found max %d\n", data2[0]);
}
