CC=gcc
LDFLAGS=$(shell python3-config --ldflags) -lcfitsio
EXT_SUFFIX=$(shell python3-config --extension-suffix) $(LDFLAGS)

all:
	gcc $(BUILD_LIB)/ffi_demo$(EXT_SUFFIX) $(LDFLAGS) -o ffi_demo

clean:
	rm -rf build ffi_demo
