CC=gcc
LDFLAGS=$(shell python3-config --ldflags) -lcfitsio
EXT_SUFFIX=$(shell python3-config --extension-suffix)

all: libarlffi

libarlffi:
	cp $(BUILD_LIB)/libarlffi$(EXT_SUFFIX) libarlffi.so

clean:
	rm -rf build libarlffi.so
