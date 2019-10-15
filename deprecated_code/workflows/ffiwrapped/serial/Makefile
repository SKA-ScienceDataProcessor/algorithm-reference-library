CC=gcc

#PLATFORM=macos
ifeq ($(PLATFORM),macos)
	CC=clang
	DYLD_FALLBACK_LIBRARY_PATH=../../..:$(DYLD_FALLBACK_LIBRARY_PATH)
endif

EXT_SUFFIX=$(shell python3-config --extension-suffix)
LIBNAME=$(shell basename libarlffi$(EXT_SUFFIX) .so)
LNAME=$(shell echo $(LIBNAME) | cut -c4- )
EXTRA_LDFLAGS=$(shell python3-config --ldflags) -lcfitsio
LDFLAGS=-L../../../ -l$(LNAME) $(EXTRA_LDFLAGS)
CFLAGS=-I../../../ffiwrappers/include
#LDFLAGS=$(shell python3-config --ldflags) -lcfitsio

EXECS=timg_serial ical_demo
all: $(EXECS)

$(EXECS): %: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: all
	LD_LIBRARY_PATH=../../..:$(LD_LIBRARY_PATH) ./timg_serial
	LD_LIBRARY_PATH=../../..:$(LD_LIBRARY_PATH) ./ical_demo

clean:
	rm -rf results timg_serial ical_demo *.o *.out
