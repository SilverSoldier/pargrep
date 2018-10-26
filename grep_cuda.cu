#include <stdio.h>
#include <cuda.h>

extern "C" {
#include "grep_cuda.h"
}

#define MAX_FILE_SIZE 1 << 30

typedef struct search_result {
  char context[30];
  int line;
  int file;
} res;

__global__ void grep_kernel(char** contents, res** results, int* results_size, const char* pattern){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int res_idx = 0;
  int valid;
  int line = 1;
  /* Read the ith file, check for pattern and write to result */
  for(int i = 0; contents[idx][i] != '\0'; i++){
	valid = 1;
	line += (contents[idx][i] == '\n');
	for(int j = 0; pattern[j] != '\0'; j++){
	  char c = contents[idx][i+j];
	  char p = pattern[j];
	  if(c == '\0')
		break;
	  valid &= (c == p);
	}
	if(valid){
	  memcpy((void*)&(results[res_idx]->context[res_idx]), (void*)&(contents[idx][i-15]), 30);
	  results[res_idx]->context[29] = '\0';
	  results[idx][res_idx].file = idx;
	  results[idx][res_idx].line = line;
	  results_size[idx]++;
	  /*   res_idx++; */
	}
  }
}

extern "C" void parallel_grep(char** file_names, char** content, int n_files, char* pattern){
  /* Copying file related data to device memory */
  char** device_contents;
  char** temp = (char**) malloc(n_files * sizeof(char*));
  cudaMalloc(&device_contents, n_files * sizeof(char*));
  for(int i = 0; i < n_files; i++){
	cudaMalloc(&temp[i], MAX_FILE_SIZE * sizeof(char));
	/* Second read of data - This is wasteful */
	cudaMemcpy(temp[i], content[i], strlen(content[i]) * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(device_contents + i, &(temp[i]), sizeof(char*), cudaMemcpyHostToDevice);
  }

  /* Copying the pattern to device memory */
  char* device_pattern;
  cudaMalloc(&device_pattern, strlen(pattern));
  cudaMemcpy(device_pattern, pattern, strlen(pattern), cudaMemcpyHostToDevice);

  /* Creating an array of array of results - one array for each thread doing computation */
  res** results;
  int* results_size;
  cudaMallocManaged(&results, n_files * sizeof(res*));
  cudaMallocManaged(&results_size, n_files * sizeof(int));
  for(int i = 0; i < n_files; i++){
	cudaMallocManaged(&(results[i]), 1000 * sizeof(res));
	results_size[i] = 0;
  }

  grep_kernel <<< 1, n_files >>> (device_contents, results, results_size, device_pattern);

  cudaDeviceSynchronize();

  for(int i = 0; i < n_files; i++)
	for(int j = 0; j < results_size[i]; j++){
	  res result = results[i][j];
	printf("%s:%d:%s", file_names[result.file], result.line, result.context);
	}
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
