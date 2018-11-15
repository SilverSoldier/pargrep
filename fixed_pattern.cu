#include <stdio.h>
#include <cuda.h>

extern "C" {
#include "fixed_pattern.h"
}

#define MAX_FILE_SIZE 1 << 30
const int CHUNK = 4 << 15;
const int MAX_CONTEXT_SIZE = 500;
const int N_RESULTS = 150;

typedef struct search_result {
  char* context;
  int line;
} res;

__global__ void fixed_pattern_kernel(char** contents, res*** results, const char* pattern, int file_no){
  int res_idx = 0;
  uint8_t valid;
  int line = 1;

  /* Local variables that keep track of the start and end of the context */
  /* TODO: out_before: needs to be initialized by going back until previous newline is found */
  int out_before = -1;
  uint8_t matched = 0;

  /* Read the ith file, check for pattern and write to result */
  char* start = &(contents[file_no][threadIdx.x * CHUNK]);
  res* result_loc = &(results[file_no][threadIdx.x][0]);
  char c;

  int i;
  for(i = 0; i < threadIdx.x * CHUNK && *(start-i) != '\n'; i++);
  out_before = -1 * i;

  for(i = 0; i < CHUNK && ((c = *(start + i)) != '\0'); i++){
	valid = 1;
	line += (c == '\n');
	if(matched && (c == '\n')){
	  /* Copy context from the previous newline character to the present character */
	  /* NOTE: Each line is only counted once - irrespective of number of occurances */
	  /* TODO: allocated only 100 bytes of space - if it exceeds, do a check and malloc as necessary */
	  memcpy((result_loc + res_idx)->context, (void*)(start + out_before+1), i - out_before - 1);
	  (result_loc + res_idx)->line = line - 1;
	  res_idx += 1;
	  matched = 0;
	}
	/* Complicated way of avoiding control divergence to keep track of the previous new line occurance */
	out_before = out_before * (c != '\n') + i * (c == '\n');

	for(int j = 0; pattern[j] != '\0'; j++){
	  char c = *(start + i + j);
	  char p = pattern[j];
	  if(c == '\0')
		break;
	  /* Could break at this stage - not sure if there will be any gain due to control divergence */
	  valid &= (c == p);
	}
	/* Need to remember whether some valid match occured on this line before - so || to not lose previous data */
	matched = matched || (valid != 0);
  }

  /* There might be some matched string still waiting to find its ending newline character */
  if(matched){
	for(; (c = *(start + i) != '\n'); i++);
	memcpy((result_loc + res_idx)->context, (void*)(start + out_before+1), i - out_before - 1);
	(result_loc + res_idx)->line = line - 1;
  }
}

extern "C" void fixed_pattern_match(char** file_names, file_info* info, int n_files, char* pattern){
  /* Copying file related data to device memory */
  char** device_contents;
  char** temp = (char**) malloc(n_files * sizeof(char*));
  cudaMalloc(&device_contents, n_files * sizeof(char*));

  cudaStream_t streams[n_files];
  for(int i = 0; i < n_files; i++){
	cudaMalloc(&temp[i], info[i].size * sizeof(char));
	cudaMemcpy(temp[i], info[i].mmap, info[i].size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_contents + i, &(temp[i]), sizeof(char*), cudaMemcpyHostToDevice);

	/* cudaHostRegister(info[i].mmap, info[i].size, 0); */
	/* cudaStreamCreate(&streams[i]); */
	/* cudaMemcpyAsync(temp[i], info[i].mmap, info[i].size, cudaMemcpyHostToDevice, streams[i]); */
	/* cudaMemcpyAsync(device_contents + i, &(temp[i]), sizeof(char*), cudaMemcpyHostToDevice, streams[i]); */
	/* Unpinning the memory */
	/* cudaHostUnregister(info[i].mmap); */
  }

  /* Copying the pattern to device memory */
  char* device_pattern;
  cudaMalloc(&device_pattern, strlen(pattern));
  cudaMemcpy(device_pattern, pattern, strlen(pattern), cudaMemcpyHostToDevice);

  /* Creating an array of array of array of results: */
  res*** results;
  int* threads_size = (int*) malloc(n_files * sizeof(int));
  /* First pointer to index the file being grepped */
  cudaMallocManaged(&results, n_files * sizeof(res**));
  for(int i = 0; i < n_files; i++){
	/* Second malloc to index the thread doing the computation */
	int n_chunks = info[i].size/CHUNK + 1;
	cudaMallocManaged(&(results[i]), n_chunks * sizeof(res*));
	threads_size[i] = n_chunks;
	for(int j = 0; j < n_chunks; j++){
	  /* Third to index the result that the thread found */
	  /* TODO: third to index the dynamic array for that result which will have a next pointer */
	  cudaMallocManaged(&(results[i][j]), N_RESULTS * sizeof(res));
	  for(int k = 0; k < N_RESULTS; k++){
		cudaMallocManaged(&(results[i][j][k].context), MAX_CONTEXT_SIZE);
	  }
	}
  }

  for(int i = 0; i < n_files; i++){
	fixed_pattern_kernel <<< 1, threads_size[i] >>> (device_contents, results, device_pattern, i);
  }

  cudaDeviceSynchronize();

  res result;
  for(int i = 0; i < n_files; i++){
	for(int j = 0; j < threads_size[i]; j++){
	  for(int k = 0; k < N_RESULTS; k++){
		result = results[i][j][k];
		if(result.line != 0)
		  printf("%s\n", result.context);
	  }
	}
  }

  cudaFree(results);
  cudaFree(device_contents);
  cudaFree(device_pattern);
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
