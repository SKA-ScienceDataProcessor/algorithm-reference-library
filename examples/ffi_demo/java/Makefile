PYCPPFLAGS=$(shell python3-config --includes)
ARLCPPFLAGS=-I../src/
ARLLDFLAGS=-L../. -larlffi
JDKHOME=/nix/store/krsx6lwd2vpp2z1p50kgidjdc781vcz5-openjdk-8u141b15/
JAVACPPFLAGS= -I${JDKHOME}/include/ -I${JDKHOME}/include/linux

default: libarljava.so arljava.jar FFIDemo.class

%.o: %.c
	gcc ${PYCPPFLAGS} ${ARLCPPFLAGS} $< -o $@  ${ARLLDFLAGS}


ffi_demo: ffi_demo.c
	gcc ${PYCPPFLAGS} ${ARLCPPFLAGS} $< -o $@  ${ARLLDFLAGS}


arljava.c: arljava.i
	mkdir -p sdp/arl	
	swig -java   -outdir sdp/arl -package sdp.java -o $@ $<


libarljava.so: arljava.c
	gcc ${JAVACPPFLAGS} -fPIC  -shared $<  -o $@ ${ARLLDFLAGS}

arljava.jar: arljava.c sdp/arl/*
	mkdir -p bin
	find sdp/arl -name "*.java" | xargs javac -g -d bin
	jar cf arljava.jar -C bin sdp


FFIDemo.class: FFIDemo.java
	javac -cp "arljava.jar" $<

clean:
	rm -r arljava.c arljava.jar bin FFIDemo.class libarljava.so sdp
