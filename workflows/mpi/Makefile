CC=mpicc

#LDFLAGS=-L../../../ -l$(LNAME) $(EXTRA_LDFLAGS)
#CFLAGS=-I../../../ffiwrappers/include
#LDFLAGS=$(shell python3-config --ldflags) -lcfitsio

LDFLAGS=
CFLAGS=

EXEC=mpi_reduce_test_c
all: $(EXEC)

$(EXECS): %: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: all
	mpiexec -np 4 -machinefile ./machinefile ./$(EXEC)
clean:
	rm -rf *.o $(EXEC)
