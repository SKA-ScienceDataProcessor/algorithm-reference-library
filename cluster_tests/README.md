
These are tests of the ability to run dask in various contexts. These are intended to be run in an initial port and
 whenever the cluster configuration has changed. The tests should be runnable from the commmand line and via SLURM
  scripts.

 - cluster_dask_test is python + Dask
 - cluster_image_test is python + Dask + ARL
 - ritoy is python + Dask
 - ritoy_numba is python + Dask + numba
 
 Known exceptions
 
 - ritoy_numba fails on P3 if the number of workers per node is greater than one. For this reason we do not currently
  recommend using numba.
 
 Tim Cornwell 12/11/2019