objects = main.o args.o nfa.o

all: $(objects)
	nvcc fixed_pattern.cu regex.cu $(objects) -o pargrep

%.o: %.c %.h
	nvcc -x c -I. -dc $< -o $@

c: main.c args.c file.h
	gcc -std=gnu99 -g main.c args.c -o main
