objects = main.o args.o

all: $(objects)
	nvcc grep_cuda.cu $(objects) -o pargrep

%.o: %.c %.h
	nvcc -x c -I. -dc $< -o $@

c: pargrep.c args.c
	gcc pargrep.c args.c -o pargrep
