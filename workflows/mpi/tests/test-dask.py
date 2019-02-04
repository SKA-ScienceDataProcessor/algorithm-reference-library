
from dask.distributed import Client
client = Client(scheduler_file='./scheduler.json')
