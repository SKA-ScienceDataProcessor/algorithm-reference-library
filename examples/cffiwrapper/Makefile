CC=g++
CFLAGS=
LFLAGS=-lpython3.6m


all: test


test: cffiwrapper.hpp main.cpp
	${CC} -o test main.cpp ${LFLAGS}


clean:
	rm test
