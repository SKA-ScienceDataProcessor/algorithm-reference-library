from arl.pipelines.dask_init import get_dask_Client

c=get_dask_Client()
print(c.scheduler_info())
exit()
