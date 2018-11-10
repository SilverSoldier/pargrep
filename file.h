#ifndef FILE_H
#define FILE_H 
#include <sys/types.h>

typedef struct file_info {
  char* mmap;
  char* file_name;
  off_t size;
} file_info;

#endif /* ifndef FILE_H */
