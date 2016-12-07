// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>
#include <stdlib.h>
#include "milli.h"

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

int main()
{
	const int N = 32;
  float *a = (float*)malloc(N*N*sizeof(float));
	float *b = (float*)malloc(N*N*sizeof(float));
	float *c = (float*)malloc(N*N*sizeof(float));

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}

  int ms = GetMicroseconds();
  //SetMilli(0,0);
	add_matrix(a, b, c, N);
  int end = GetMicroseconds();
  printf("CPU, time taken in microseconds: %i\n", end - ms);
  /*
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}
  */
  free(a);
  free(b);
  free(c);
}
