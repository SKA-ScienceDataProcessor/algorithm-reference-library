#!/usr/bin/env python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

def fn_sum(buffer_a, buffer_b, t):
    tc = MPI._typecode(t) # map MPI datatype -> Python typecode
    array_a = np.frombuffer(buffer_a, dtype=tc)
    array_b = np.frombuffer(buffer_b, dtype=tc)
    array_b += array_a

op_sum = MPI.Op.Create(fn_sum, commute=True)

data = np.empty(5, dtype='i')
data.fill(comm.rank)
result = np.empty_like(data)
#comm.Allreduce(data, result, op=op_sum)
result=comm.reduce(data, op=op_sum)

print(result)
assert np.allclose(result, sum(range(comm.size)))
