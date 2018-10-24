#include <stdio.h>
#include <cuda.h>

extern "C" {
#include "grep_cuda.h"
}

#define MAX_FILE_SIZE 1 << 30

__global__ void test(int* A){
  A[threadIdx.x + blockIdx.x * blockDim.x] = 10;
}

extern "C" void parallel_grep(){
  int* host_A = (int*) malloc(1000 * sizeof(int));
  int* device_A;
  cudaMalloc((void**)&device_A, 1000 * sizeof(int));

  cudaMemcpy(device_A, host_A, 1000 * sizeof(int), cudaMemcpyHostToDevice);

  test <<< 1000, 1 >>> (device_A);

  cudaMemcpy(host_A, device_A, 1000 * sizeof(int), cudaMemcpyDeviceToHost);

  for(int i = 0; i < 100; i++)
	if(host_A[i] != 10){
	  printf("Check failed.\n");
	  free(host_A);
	  cudaFree(device_A);
	  return;
	}

  printf("Check passed.\n");
  free(host_A);
  cudaFree(device_A);
}
