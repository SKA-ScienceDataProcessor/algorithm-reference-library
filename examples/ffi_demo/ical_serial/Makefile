CC=gcc
LDFLAGS=-L../. -larlffi

all: ical_serial

ical_serial:
	$(CC) ical_demo.c $(LDFLAGS) -o ical_serial

run: all
	LD_LIBRARY_PATH=../.:$(LD_LIBRARY_PATH) ./ical_serial

clean:
	rm -rf results ical_serial 
