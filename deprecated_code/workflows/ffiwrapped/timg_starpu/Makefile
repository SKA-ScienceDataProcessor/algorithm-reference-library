CC      = gcc
CFLAGS  = $(shell pkg-config --cflags starpu-1.2)
LDFLAGS = $(shell pkg-config --libs starpu-1.2) -L../. -larlffi -L/home/at748/Downloads/cfitsio/lib -lcfitsio -lcurl -lm

all: timg_starpu

timg_starpu: timg_starpu.so
	gcc timg_starpu.o timg_pu_routines.o timg_pu_codelets.o $(LDFLAGS) -o timg_starpu 

timg_starpu.so: timg_pu_routines.o
	gcc -c timg_starpu.c $(CFLAGS) -o timg_starpu.o

timg_pu_routines.o: timg_pu_codelets.o
	gcc -c timg_pu_routines.c $(CFLAGS) -o timg_pu_routines.o

timg_pu_codelets.o:
	gcc -c timg_pu_codelets.c $(CFLAGS) -o timg_pu_codelets.o

run: all
	LD_LIBRARY_PATH=../.:$(LD_LIBRARY_PATH) ./timg_starpu

clean:
	rm -rf results timg_starpu *.o
