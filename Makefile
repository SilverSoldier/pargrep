c = args.o 

cuda = regex.o fixed_pattern.o nfa.o

main = main.o

all: $(c) $(cuda) $(main)
	nvcc $(c) $(cuda) $(main) -o pargrep

$(main): %.o: %.c
	nvcc -x c -I. -dc $< -o $@

$(c): %.o: %.c %.h
	nvcc -x c -I. -dc $< -o $@

$(cuda): %.o: %.cu %.h
	nvcc -c -I. -dc $< -o $@

cpu: main.c args.c file.h
	gcc -std=gnu99 -g main.c args.c -o main

clean:
	rm *.o
