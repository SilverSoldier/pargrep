all: pargrep.c args.c
	gcc pargrep.c args.c -o pargrep
