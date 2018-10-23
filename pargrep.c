#include <stdlib.h>
#include <stdio.h>
#include "args.h"

int main(int argc, char *argv[]) {
  struct arguments arguments;
  parse_options(argc, argv, &arguments);
  return 0;
}
