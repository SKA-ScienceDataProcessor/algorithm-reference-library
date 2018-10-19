from distributed import Client
from dask import delayed
import time

if __name__ == "__main__":
    
    c = Client()
    addr = c.scheduler_info()['address']
    services = c.scheduler_info()['services']
    if 'bokeh' in services.keys():
        bokeh_addr = 'http:%s:%s' % (addr.split(':')[1], services['bokeh'])
        print('Diagnostic pages available on port %s' % bokeh_addr)
    
    work = [delayed(time.sleep)(5.0 + i / 100.0) for i in range(100)]
    
    c.compute(work, sync=True)
    c.close()
    
    print('Finished')