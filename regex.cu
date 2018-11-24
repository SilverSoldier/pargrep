#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

extern "C" {
#include "regex.h"
#include "nfa.h"
}

#define MAX_FILE_SIZE 1 << 30
const int CHUNK = 1 << 14;
const int MAX_CONTEXT_SIZE = 500;
const int N_RESULTS = 50;
const int MAX_THREADS_PER_BLOCK = 1024;

typedef struct search_result {
  char* context;
  int line;
} res;

__global__ void regex_kernel(char** contents, res*** results, const re_t __restrict__ pattern, int file_no){
  int res_idx = 0;
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
  for(i = 0; i < threadIdx.x * CHUNK && *(start-i) != '\n' && *(start-i) != '\0'; i++);
  out_before = -1 * i - (i == 0);

  for(i = 0; i < CHUNK && ((c = *(start + i)) != '\0'); i++){
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
	matched += re_matchp(pattern, (start + i));
  }

  /* There might be some matched string still waiting to find its ending newline character */
  if(matched){
	for(; (c = *(start + i) != '\n' && c != '\0'); i++);
	memcpy((result_loc + res_idx)->context, (void*)(start + out_before+1), i - out_before - 1);
	(result_loc + res_idx)->line = line - 1;
  }
}

extern "C" void regex_match(char** file_names, file_info* info, int n_files, char* pattern){

  struct timeval start, end;
  gettimeofday(&start, NULL);

  /* re_print(re_pattern); */

  /* Copying file related data to device memory */
  char** device_contents;
  char** temp = (char**) malloc(n_files * sizeof(char*));
  cudaMalloc(&device_contents, n_files * sizeof(char*));

  cudaStream_t streams[n_files];
  for(int i = 0; i < n_files; i++){
	/* cudaMalloc(&temp[i], MAX_FILE_SIZE * sizeof(char)); */
	/* cudaMemcpy(temp[i], info[i].mmap, info[i].size, cudaMemcpyHostToDevice); */
	/* cudaMemcpy(device_contents + i, &(temp[i]), sizeof(char*), cudaMemcpyHostToDevice); */

	cudaStreamCreate(&streams[i]);
	cudaHostRegister(info[i].mmap, info[i].size, 0);
	cudaMallocHost(&temp[i], info[i].size);
	cudaMemcpyAsync(temp[i], info[i].mmap, info[i].size, cudaMemcpyHostToDevice, streams[i]);
	cudaMemcpyAsync(device_contents + i, &(temp[i]), sizeof(char*), cudaMemcpyHostToDevice, streams[i]);
  }

  re_t re_pattern = re_compile(pattern);
  gettimeofday(&end, NULL);

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
	  cudaMallocManaged(&(results[i][j]), N_RESULTS * sizeof(res));
	  for(int k = 0; k < N_RESULTS; k++){
		cudaMallocManaged(&(results[i][j][k].context), MAX_CONTEXT_SIZE);
	  }
	}
  }

  for(int i = 0; i < n_files; i++){
	if(threads_size[i] > MAX_THREADS_PER_BLOCK){
	  int n_blocks = threads_size[i]/MAX_THREADS_PER_BLOCK + 1;
	  regex_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK, 0, streams[i] >>> (device_contents, results, re_pattern, i);
	}
	else{
	  regex_kernel <<< 1, threads_size[i], 0, streams[i] >>> (device_contents, results, re_pattern, i);
	}
	/* Unpinning the memory */
	cudaHostUnregister(info[i].mmap);
  }

  cudaDeviceSynchronize();


  /* printf("%s\n", cudaGetErrorString(cudaPeekAtLastError())); */
  /* printf("%d\n", results[0][0][0].line); */

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
  cudaFree(re_pattern);

  printf("Kernel: %f\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/(double)1000);
}

