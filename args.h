struct arguments {
  char* pattern;
  char** files;
  int regex;
};

void parse_options(int argc, char* argv[], struct arguments* arguments);
