CC=gcc
EXT_SUFFIX=$(shell python3-config --extension-suffix)
LIBNAME=$(shell basename libarlffi$(EXT_SUFFIX) .so)
LNAME=$(shell echo $(LIBNAME) | cut -c4- )
LDFLAGS=-L../../../ -l$(LNAME)
CFLAGS=-I../../../ffiwrappers/include
#LDFLAGS=$(shell python3-config --ldflags) -lcfitsio


all: timg_serial ical_demo

timg_serial:
	$(CC) $(CFLAGS) timg_serial.c $(LDFLAGS) -o timg_serial

ical_serial:
	$(CC) $(CFLAGS) ical_demo.c $(LDFLAGS) -o ical_demo

run: all
	LD_LIBRARY_PATH=../../..:$(LD_LIBRARY_PATH) ./timg_serial 

clean:
	rm -rf results timg_serial ical_demo
