/*
 * Image filter in OpenCL
 */

#define KERNELSIZE 2
#define BLOCKS 16
#define PRELOAD_PIXELS 2
#define PIXEL_SIZE 3

__kernel void filter(
         __global unsigned char *image,
         __global unsigned char *out,
         const unsigned int n,
         const unsigned int m)
{
  // Local storage
  __local unsigned char *local_pixel_storage[BLOCKS*PRELOAD_PIXELS*PIXEL_SIZE];

  // Local ids
  unsigned int local_y = get_local_id(1) % 512;
  unsigned int local_x = get_local_id(0) % 512;
  unsigned int local_size = get_local_size(0);

  // Global ids
  unsigned int i = get_global_id(1) % 512;
  unsigned int j = get_global_id(0) % 512;

  // temporary variable for storing starting index of a local pixel
  uint local_start_index;

  // Load pixels into local memory
  for (uint idx = 0; idx < PRELOAD_PIXELS; ++idx)
    {
      // local index which we store the pixel to (column,row)
      local_start_index = (local_y + idx)*(local_size * KERNELSIZE)+(local_x + idx);
      local_start_index *= PIXEL_SIZE;
      // global index which we load the pixel from (column,row)
      uint global_start_index = (i + idx)*n+(j + idx);
      global_start_index *= PIXEL_SIZE;

      // Store colors for pixel
      for (uint id_color = 0; id_color < PIXEL_SIZE; ++id_color)
        {
          local_pixel_storage[local_start_index + id_color] = ( 
            image[global_start_index + id_color]); 
        }
    }
  
  // pixel indexes
  int k, l;
  __constant int divby = (2*KERNELSIZE+1)*(2*KERNELSIZE+1);
  unsigned int sumx, sumy, sumz;

  // Ensure all the local pixels are loaded in local memory before computing
  barrier(CLK_LOCAL_MEM_FENCE);

  // If we are inside the image
  if (j < n && i < m)
  {
    // if we are in our local scope and not edge
    if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE)
    {
      // Filter kernel
      sumx=0;sumy=0;sumz=0;
      for(k=-KERNELSIZE;k<=KERNELSIZE;k++)
        for(l=-KERNELSIZE;l<=KERNELSIZE;l++)
        {
          local_start_index = (local_y + k)*(local_size * KERNELSIZE)+(local_x + l);
          local_start_index *= PIXEL_SIZE;
          sumx += local_pixel_storage[local_start_index + 0];
          sumy += local_pixel_storage[local_start_index + 1];
          sumz += local_pixel_storage[local_start_index + 2];
        }
      out[(i*n+j)*3+0] = sumx/divby;
      out[(i*n+j)*3+1] = sumy/divby;
      out[(i*n+j)*3+2] = sumz/divby;
    }
    else
    // Edge pixels are not filtered
    {
      out[(i*n+j)*3+0] = image[(i*n+j)*3+0];
      out[(i*n+j)*3+1] = image[(i*n+j)*3+1];
      out[(i*n+j)*3+2] = image[(i*n+j)*3+2];
    }
  }
}
