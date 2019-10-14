CC      = gcc
CFLAGS  = -ggdb -O0 -Wall $(shell pkg-config --cflags starpu-1.2) -I/opt/algorithm-reference-library/ffiwrappers/include
LDFLAGS = $(shell pkg-config --libs starpu-1.2) -L../../../ -L../. -larlffi.cpython-36m-x86_64-linux-gnu -larlffi -L/usr/lib64 -lcfitsio -lcurl -lm

all: ical_starpu ical_starpu_implicit

ical_starpu: ical_starpu.so
	gcc ical_starpu.o ical_pu_routines.o ical_pu_codelets.o $(LDFLAGS) -o ical_starpu 

ical_starpu_implicit: ical_starpu_implicit.so
	gcc ical_starpu_implicit.o ical_pu_routines.o ical_pu_codelets.o $(LDFLAGS) -o ical_starpu_implicit 

ical_starpu.so: ical_pu_routines.o
	gcc -c ical_starpu.c $(CFLAGS) -o ical_starpu.o

ical_starpu_implicit.so: ical_pu_routines.o
	gcc -c ical_starpu_implicit.c $(CFLAGS) -o ical_starpu_implicit.o

ical_pu_routines.o: ical_pu_codelets.o
	gcc -c ical_pu_routines.c $(CFLAGS) -o ical_pu_routines.o

ical_pu_codelets.o:
	gcc -c ical_pu_codelets.c $(CFLAGS) -o ical_pu_codelets.o

run: all
	LD_LIBRARY_PATH=../.:$(LD_LIBRARY_PATH) ./ical_starpu

clean:
	rm -rf results ical_starpu *.o
