#include <stdio.h>
#include <cuda.h>

extern "C" {
#include "grep_cuda.h"
}

#define MAX_FILE_SIZE 1 << 30

__global__ void grep_kernel(char** contents, int* count){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  /* Let's count lines */
  count[idx] = 0;
  for(int i = 0; contents[idx][i] != '\0'; i++){
	if(contents[idx][i] == '\n')
	  count[idx]++;
  }
}

extern "C" void parallel_grep(char** content, int n_files, char* pattern){
  /* Each thread takes a file */
  char** device_contents;
  char** temp = (char**) malloc(n_files * sizeof(char*));
  cudaMalloc(&device_contents, n_files * sizeof(char*));
  for(int i = 0; i < n_files; i++){
	cudaMalloc(&temp[i], MAX_FILE_SIZE * sizeof(char));
	/* Second read of data - This is wasteful */
	cudaMemcpy(temp[i], content[i], strlen(content[i]) * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(device_contents + i, &(temp[i]), sizeof(char*), cudaMemcpyHostToDevice);
  }

  int* count_host = (int*) calloc(n_files, sizeof(int));
  int* count_device;
  cudaMalloc((void**)&count_device, n_files * sizeof(int));

  cudaMemcpy(count_device, count_host, n_files * sizeof(int), cudaMemcpyHostToDevice);
  
  grep_kernel <<< 1, n_files >>> (device_contents, count_device);

  cudaMemcpy(count_host, count_device, n_files * sizeof(int), cudaMemcpyDeviceToHost);

  for(int i = 0; i < n_files; i++)
	printf("%d ", count_host[i]);
  printf("\n");
}

__global__ void test_kernel(int* A){
  A[threadIdx.x + blockIdx.x * blockDim.x] = 10;
}

extern "C" void test(){
  int* host_A = (int*) malloc(1000 * sizeof(int));
  int* device_A;
  cudaMalloc((void**)&device_A, 1000 * sizeof(int));

  cudaMemcpy(device_A, host_A, 1000 * sizeof(int), cudaMemcpyHostToDevice);

  test_kernel <<< 1, 1000 >>> (device_A);

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
