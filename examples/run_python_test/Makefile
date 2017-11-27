C		= gcc
LDFLAGS		=$(shell python3-config --ldflags)
CFLAGS		=$(shell python3-config --cflags)


OBJ = run_python.o

run_python: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o run_python $(LDFLAGS)

all:run_python

.PHONY: clean
clean: 
	rm -f *~ *.o run_python
