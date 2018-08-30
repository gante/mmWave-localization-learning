GCC = g++
CFLAGS = -O3 -std=c++11 -Wall -march=native -mtune=native

all: main cleanup
 
main: main.o
	$(GCC) $(CLINKFLAGS) -o data_preprocessing  main.o
	
main.o: main.cpp general_includes.hpp
	$(GCC) $(CFLAGS) -c main.cpp 
	
cleanup:
	#WINDOWS: del *.o
	rm *.o
	
clean:
	#WINDOWS del *.o
	rm *.o
