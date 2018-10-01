from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'a': 7, 'b': 3.14}
    data = [3,4]
    data = comm.scatter(data, root=0)
    print(data)
else:
    data=None
    data = comm.scatter(data, root=0)
    print(data)

