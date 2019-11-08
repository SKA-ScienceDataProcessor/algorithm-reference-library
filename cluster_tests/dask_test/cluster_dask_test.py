import random
import time
import os

from dask.distributed import Client

if __name__ == '__main__':
    
    snooze = 1.0

    def inc(x):
        time.sleep(snooze * random.random())
        return x + 1
    
    def dec(x):
        time.sleep(snooze * random.random())
        return x - 1
    
    def add(x, y):
        time.sleep(snooze * random.random())
        return x + y
    
    print("Starting cluster_dask_test")
    scheduler = os.getenv('ARL_DASK_SCHEDULER', None)
    if scheduler is not None:
        print("Creating Dask Client using externally defined scheduler")
        client = Client(scheduler)
    else:
        print("Using Dask on this computer")
        client = Client(threads_per_worker=1, n_workers=8)
        
    import dask
    inc = dask.delayed(inc)
    dec = dask.delayed(dec)
    add = dask.delayed(add)

    zs = []
    for i in range(256):
        x = inc(i)
        y = dec(x)
        z = add(x, y)
        zs.append(z)

    zs = dask.persist(*zs)
    L = zs
    while len(L) > 1:
        new_L = []
        for i in range(0, len(L), 2):
            lazy = add(L[i], L[i + 1])  # add neighbors
            new_L.append(lazy)
        L = new_L  # swap old list for new

    result = dask.compute(L, sync=True)
    assert result[0][0] == 65536
    print("Finished cluster_dask_test correctly")
    
    client.close()
    
    exit(0)


    
