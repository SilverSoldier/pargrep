objects = main.o args.o

all: $(objects)
	nvcc grep_cuda.cu $(objects) -o pargrep

%.o: %.c %.h
	nvcc -x c -I. -dc $< -o $@

c: main.c args.c file.h
	gcc -std=gnu99 -g main.c args.c file.h -o main
