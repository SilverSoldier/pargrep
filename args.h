struct arguments {
  char* pattern;
  char** files;
  int recursive;
};

void parse_options(int argc, char* argv[], struct arguments* arguments);
