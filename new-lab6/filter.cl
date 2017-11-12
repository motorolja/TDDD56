/*
 * Image filter in OpenCL
 */

#define KERNELSIZE 2
#define BLOCK_DIM 16
#define PIXEL_SIZE 3

__kernel void filter(
         __global unsigned char *image,
         __global unsigned char *out,
         const unsigned int n,
         const unsigned int m)
{
  unsigned int DIM = KERNELSIZE*2+1;
  // Local ids
  unsigned int local_y = get_local_id(1) % 512; // threadIdx CUDA
  unsigned int local_x = get_local_id(0) % 512; // threadIdx CUDA
  //unsigned int local_size = get_local_size(0); // blockDim CUDA

  // Local storage all required storage for the locally cached pixels
  __local unsigned char local_pixel_storage[(BLOCK_DIM+2*KERNELSIZE)*(BLOCK_DIM+2*KERNELSIZE)*PIXEL_SIZE];

  // Global ids
  unsigned int global_y = get_global_id(1) % 512;
  unsigned int global_x = get_global_id(0) % 512;

  // pixel indexes
  int k, l;

  int divby = DIM*DIM;
  unsigned int sumx, sumy, sumz;
  // temporary variable for storing global starting index of a global pixel
  unsigned int global_start_index;
  // temporary variable for storing starting index of a local pixel
  unsigned int local_start_index;


  // If we are inside the range of filtering
  if (global_y >= KERNELSIZE && global_y < m-KERNELSIZE && global_x >= KERNELSIZE && global_x < n-KERNELSIZE)
  {
    // Load the edge pixels into local memory, for all local threads
    // NOTE: local storage starts with pixels that are -2 in x-axis and -2 in y-axis

    // lower left
    local_start_index = ((local_y + 0)*(BLOCK_DIM + KERNELSIZE*2)+(local_x + 0))*PIXEL_SIZE;
    global_start_index = ((global_y + 0)*n+(global_x + 0))*PIXEL_SIZE;

    // Store colors for pixel
    local_pixel_storage[local_start_index + 0] = image[global_start_index + 0];
    local_pixel_storage[local_start_index + 1] = image[global_start_index + 1];
    local_pixel_storage[local_start_index + 2] = image[global_start_index + 2];

    // lower right
    local_start_index = ((local_y + 0)*(BLOCK_DIM + KERNELSIZE*2)+(local_x + 2*KERNELSIZE))*PIXEL_SIZE;
    global_start_index = ((global_y + 0)*n+(global_x + 2*KERNELSIZE))*PIXEL_SIZE;

    // Store colors for pixel
    local_pixel_storage[local_start_index + 0] = image[global_start_index + 0];
    local_pixel_storage[local_start_index + 1] = image[global_start_index + 1];
    local_pixel_storage[local_start_index + 2] = image[global_start_index + 2];

    // upper left
    local_start_index = ((local_y + 2*KERNELSIZE)*(BLOCK_DIM + KERNELSIZE*2)+(local_x + 0))*PIXEL_SIZE;
    global_start_index = ((global_y + 2*KERNELSIZE)*n+(global_x + 0))*PIXEL_SIZE;

    // Store colors for pixel
    local_pixel_storage[local_start_index + 0] = image[global_start_index + 0];
    local_pixel_storage[local_start_index + 1] = image[global_start_index + 1];
    local_pixel_storage[local_start_index + 2] = image[global_start_index + 2];

    // upper right
    local_start_index = ((local_y + KERNELSIZE*2)*(BLOCK_DIM + KERNELSIZE*2)+(local_x + KERNELSIZE*2))*PIXEL_SIZE;
    global_start_index = ((global_y + KERNELSIZE*2)*n+(global_x + KERNELSIZE*2))*PIXEL_SIZE;

    // Store colors for pixel
    local_pixel_storage[local_start_index + 0] = image[global_start_index + 0];
    local_pixel_storage[local_start_index + 1] = image[global_start_index + 1];
    local_pixel_storage[local_start_index + 2] = image[global_start_index + 2];

    // Ensure all the local pixels are loaded in local memory before computing
    barrier(CLK_LOCAL_MEM_FENCE);

    // Filter kernel
		sumx=0;sumy=0;sumz=0;
    for(k=-KERNELSIZE;k<=KERNELSIZE;k++)
    {
      for(l=-KERNELSIZE;l<=KERNELSIZE;l++)
      {
        local_start_index = ((local_y + KERNELSIZE + k)*(BLOCK_DIM + KERNELSIZE*2)+(local_x + KERNELSIZE + l))*PIXEL_SIZE;
        sumx += local_pixel_storage[local_start_index + 0];
        sumy += local_pixel_storage[local_start_index + 1];
        sumz += local_pixel_storage[local_start_index + 2];
      }
    }
    global_start_index = (global_y*n+global_x)*PIXEL_SIZE;
    out[global_start_index+0] = sumx/divby;
    out[global_start_index+1] = sumy/divby;
    out[global_start_index+2] = sumz/divby;
  }
  else
  // Edge pixels are not filtered
  {
    global_start_index = (global_y*n+global_x)*PIXEL_SIZE;
    out[global_start_index+0] = image[global_start_index+0];
    out[global_start_index+1] = image[global_start_index+1];
    out[global_start_index+2] = image[global_start_index+2];
  }
}
