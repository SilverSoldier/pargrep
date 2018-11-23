#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include "args.h"
#include "fixed_pattern.h"
#include "regex.h"

#define MAX_FILE_SIZE 1 << 30

int main(int argc, char *argv[]) {
  struct timeval start, end;

  struct arguments arguments;
  parse_options(argc, argv, &arguments);

  file_info* files;
  int file_count = 0;

  for(int j = 0; arguments.files[j]; j++)
	file_count++;

  /* Load the files into memory */
  files = (file_info*) malloc(file_count * sizeof(file_info));
  for(int j = 0; j < file_count; j++){
	FILE* fp = fopen(arguments.files[j], "r");
	int fd = fileno(fp);
	struct stat sb;
	fstat(fd, &sb);
	files[j].mmap = (char*) mmap(0, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
	files[j].size = sb.st_size;
	fclose(fp);
  }

  gettimeofday(&start, NULL);
  if(arguments.fixed == 1){
	/* Fixed string */
	fixed_pattern_match(arguments.files, files, file_count, arguments.pattern);
  } else {
	regex_match(arguments.files, files, file_count, arguments.pattern);
  }
  gettimeofday(&end, NULL);

  printf("%f\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/(double)1000);

  return 0;
}
