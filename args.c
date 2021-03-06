#include <argp.h>
#include "args.h"

const char* pargrep_program_version = "pargrep-0.1";

static char doc[] =
" Parallel grep program.\
	Currently supports only fixed string inputs";

static char args_doc[] = "PATTERN [FILE]...";

static struct argp_option options[] = {
  { "regex", 'r', 0, 0, "Regex pattern." },
  { "fixed", 'f', 0, 0, "Fixed pattern." },
  { 0 }
};

static error_t parse_opt(int key, char* arg, struct argp_state *state){
  struct arguments *arguments = state->input;

  switch(key){
	case 'r':
	  arguments->regex = 1;
	  break;

	case 'f':
	  arguments->fixed = 1;
	  break;

	case ARGP_KEY_NO_ARGS:
	  argp_usage(state);
	  break;

	case ARGP_KEY_ARG:
	  arguments->pattern = arg;

	  /* Consume rest of the arguments */
	  arguments->files = &state->argv[state->next];
	  state->next = state->argc;
	  break;

	case ARGP_KEY_END:
	  if (state->arg_num < 2)
		/* Not enough arguments. */
		argp_usage(state);
	  break;

	default:
	  return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

void parse_options(int argc, char* argv[], struct arguments* arguments){
  argp_parse(&argp, argc, argv, 0, 0, arguments);
}
