GCC = g++
CFLAGS = -O3 -std=c++11 -Wall -march=native -mtune=native

all: main cleanup
 
main: main.o
	$(GCC) $(CLINKFLAGS) -o data_preprocessing.exe  main.o
	
main.o: main.cpp general_includes.hpp
	$(GCC) $(CFLAGS) -c main.cpp 
	
cleanup:
	del *.o
	
clean:
	del *.o