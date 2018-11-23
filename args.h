struct arguments {
  char* pattern;
  char** files;
  int regex;
  int fixed;
};

void parse_options(int argc, char* argv[], struct arguments* arguments);
