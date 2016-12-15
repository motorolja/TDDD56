/*
 * Image filter in OpenCL
 */

#define KERNELSIZE 2
#define BLOCKS 16

__kernel void filter(__global unsigned char *image, __global unsigned char *out, const unsigned int n, const unsigned int m)
{
  // Local storage
  __local unsigned char local_storage[ BLOCKS + KERNELSIZE*2];
  // local ids
  unsigned int local_y = get_local_id(1) % 512;
  unsigned int local_x = get_local_id(0) % 512;
  unsigned int local_size = get_local_size(0);

  unsigned int global_y = get_global_id(1) % 512;
  unsigned int global_x = get_global_id(0) % 512;
  int k = 0, l = 0;
  unsigned int sumx, sumy, sumz;

	int divby = (2*KERNELSIZE+1)*(2*KERNELSIZE+1);

  uint tmp_local, tmp_global;

  // save two pixels
  for (int s = 0; s < 2; ++s)
    {
      for (uint i = 0; i < 3; ++i)
      {
        tmp_local = (local_y + k) * (local_size + KERNELSIZE) + local_x + l;
        tmp_local *=3;
        tmp_local += i;
        tmp_global = ((global_y + k) * n + global_x + l) * 3 + i;
        local_storage[tmp_local] = image[tmp_global];
      }
      // preload next pixel
      k += KERNELSIZE;
      l += KERNELSIZE;
    }
  // reset x and y coords l,k
  k = 0;
  l = 0;

	if (global_x < n && global_y < m) // If inside image
	{
		if (global_x >= KERNELSIZE && global_x < m-KERNELSIZE
        && global_y >= KERNELSIZE && global_y < n-KERNELSIZE)
		{
		// Filter kernel
			sumx=0;sumy=0;sumz=0;
			for(;k <= KERNELSIZE; k++)
				for(; l <= KERNELSIZE;l++)
				{
          tmp_local = (local_y + k)
            * (local_size + KERNELSIZE)
            + (local_x + l)
            * 3;
          // RGB
					sumx += image[tmp_local];
					sumy += image[tmp_local + 1];
					sumz += image[tmp_local + 2];
				}
      // RGB
			out[(global_y * n + global_x)*3+0] = sumx/divby;
			out[(global_y * n + global_x)*3+1] = sumy/divby;
			out[(global_y * n + global_x)*3+2] = sumz/divby;
		}
		else
		// Edge pixels are not filtered
		{
      tmp_local = (local_y + k) * (local_size + KERNELSIZE) + local_x + l;
      tmp_local *=3;
   		out[(global_y * n + global_x) * 3 + 0] = local_storage[tmp_local + 0];
			out[(global_y * n + global_x) * 3 + 1] = local_storage[tmp_local + 1];
			out[(global_y * n + global_x) * 3 + 2] = local_storage[tmp_local + 2];
		}
	}

}
