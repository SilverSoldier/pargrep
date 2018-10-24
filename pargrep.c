#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "args.h"
#include "grep_cuda.h"

#define MAX_FILE_SIZE 1 << 30

int main(int argc, char *argv[]) {
  struct arguments arguments;
  parse_options(argc, argv, &arguments);

  char** files;
  int file_count;

  printf("Parsed arguments\n");
  for(int j = 0; arguments.files[j]; j++)
	file_count++;

  /* Load the files into memory */
  files = (char**) malloc(file_count * sizeof(char*));
  for(int j = 0; j < file_count; j++){
	/* files[j] = (char*) malloc(MAX_FILE_SIZE); */
	/* FILE* fp = fopen(arguments.files[j], "r"); */
	/* fread((void*) files[j], sizeof(char), MAX_FILE_SIZE, fp); */
	/* fclose(fp); */
  }

  /* Creation and initialization of device related variables */

  // Copy files into device memory

  // Call the GPU handler
  printf("Calling GPU handler\n");
  parallel_grep();


  return 0;
}
