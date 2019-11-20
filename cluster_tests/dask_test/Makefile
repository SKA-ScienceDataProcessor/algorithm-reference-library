
PYFILE        = cluster_dask_test.py

.PHONY: clean
clean:
	\rm -rf *.log slurm*.out dask-worker-space hostfile.*

.PHONY: test
test: ${PYFILE}
	python ${PYFILE}





