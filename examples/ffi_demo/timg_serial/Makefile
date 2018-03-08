CC=gcc
LDFLAGS=-L../. -larlffi

all: timg_serial

timg_serial: timg_serial.c
	$(CC) timg_serial.c $(LDFLAGS) -o timg_serial

run: all
	LD_LIBRARY_PATH=../.:$(LD_LIBRARY_PATH) ./timg_serial

clean:
	rm -rf results timg_serial 
