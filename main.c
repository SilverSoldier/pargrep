#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "args.h"
#include "fixed_pattern.h"
#include "regex.h"
#include "nfa.h"

#define MAX_FILE_SIZE 1 << 30

int main(int argc, char *argv[]) {
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

  if(arguments.regex == 0){
	/* Fixed string */
	/* fixed_pattern_match(arguments.files, files, file_count, arguments.pattern); */
  } else {
	/* Regex pattern */
	char *post;
	State *start, *matchstate;

	re2post(arguments.pattern);

	if(post == NULL){
	  fprintf(stderr, "bad regexp %s\n", argv[1]);
	  return 1;
	}

	start = post2nfa(post, matchstate);
	if(start == NULL){
	  fprintf(stderr, "error in post2nfa %s\n", post);
	  return 1;
	}

	regex_match(arguments.files, files, file_count, start);

  }

  return 0;
}
