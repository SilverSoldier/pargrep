#include <sys/types.h>

typedef struct file_info {
  char* mmap;
  char* file_name;
  off_t size;
} file_info;
