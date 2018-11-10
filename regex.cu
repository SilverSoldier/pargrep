#include <stdio.h>
#include <cuda.h>

extern "C" {
#include "regex.h"
#include "nfa_device.h"
}

#define MAX_FILE_SIZE 1 << 30
const int CHUNK = 5000;
const int MAX_CONTEXT_SIZE = 500;

typedef struct search_result {
  char* context;
  int line;
} res;

__global__ void regex_kernel(char** contents, res*** results, State* pattern, int file_no){
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

extern "C" void regex_match(char** file_names, file_info* info, int n_files, State* pattern){
  /* Copying file related data to device memory */
  char** device_contents;
  char** temp = (char**) malloc(n_files * sizeof(char*));
  cudaMalloc(&device_contents, n_files * sizeof(char*));

  cudaStream_t streams[n_files];
  for(int i = 0; i < n_files; i++){
	cudaHostRegister(info[i].mmap, info[i].size, 0);
	cudaMalloc(&temp[i], MAX_FILE_SIZE * sizeof(char));
	/* cudaMemcpy(temp[i], info[i].mmap, info[i].size, cudaMemcpyHostToDevice); */
	/* cudaMemcpy(device_contents + i, &(temp[i]), sizeof(char*), cudaMemcpyHostToDevice); */
	cudaStreamCreate(&streams[i]);
	/* cudaMallocHost(&(info[i].mmap), (size_t) info[i].size); */
	/* cudaMallocHost(&temp[i], info[i].size); */
	cudaMemcpyAsync(temp[i], info[i].mmap, info[i].size, cudaMemcpyHostToDevice, streams[i]);
	cudaMemcpyAsync(device_contents + i, &(temp[i]), sizeof(char*), cudaMemcpyHostToDevice, streams[i]);
  }

  /* Copying the pattern to device memory */
  State* device_pattern;
  cudaMalloc(&device_pattern, sizeof(State));
  cudaMemcpy(device_pattern, pattern, sizeof(State), cudaMemcpyHostToDevice);

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
	  /* TODO: Fourth to index the dynamic array for that result which will have a next pointer */
	  cudaMallocManaged(&(results[i][j]), 50 * sizeof(res));
	  for(int k = 0; k < 50; k++){
		cudaMallocManaged(&(results[i][j][k].context), MAX_CONTEXT_SIZE);
	  }
	}
  }

  for(int i = 0; i < n_files; i++){
	regex_kernel <<< 1, threads_size[i], 0, streams[i] >>> (device_contents, results, pattern, i);
	/* Unpinning the memory */
	cudaHostUnregister(info[i].mmap);
  }

  cudaDeviceSynchronize();

  res result;
  for(int i = 0; i < n_files; i++){
	for(int j = 0; j < threads_size[i]; j++){
	  for(int k = 0; k < 20; k++){
		result = results[i][j][k];
		if(result.line != 0)
		  printf("%s\n", result.context);
	  }
	}
  }
}
