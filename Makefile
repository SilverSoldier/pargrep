all: pargrep.c grep_cuda.cu args.c
	gcc pargrep.c args.c -c
	nvcc grep_cuda.cu pargrep.o args.o -o pargrep

c: pargrep.c args.c
	gcc pargrep.c args.c -o pargrep
