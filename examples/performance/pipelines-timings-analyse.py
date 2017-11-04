
import os
import sys
sys.path.append(os.path.join('..', '..'))

import matplotlib.pyplot as plt

import logging

import numpy

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def analzye_scaling(filename):
    """
    Performs standard analysis of output from pipelines-timings.py. We will add to this as necessary.
    
    run as:
        python pipelines-timings-analyse.py *.csv
    :results: results dictionary
    """
    context='scaling'
    
    print("Analyzing %s for file %s " % (context, filename))
    results = read_results(filename, context)
    plt.clf()
    threads = numpy.array([int(r['threads_per_worker']) for r in results])
    n_workers = numpy.array([int(r['n_workers']) for r in results])
    nnodes = numpy.array([int(r['nnodes']) for r in results])

    print(threads)
    highest = 0.0
    for n_node in numpy.unique(nnodes):
        for label in ['time overall', 'time predict', 'time invert', 'time psf invert', 'time ICAL']:
            x = list()
            y = list()
            for i, row in enumerate(results):
                if n_node == int(row['nnodes']):
                    x.append(float(row['n_workers']))
                    y.append(float(row[label]))
            legend = '%s %d nodes' % (label.replace('time ', ''), n_node)
            plt.loglog(x, y, label=legend)
            highest = max(highest, numpy.max(y))

    x = n_workers
    y = highest/numpy.array(x)
    plt.loglog(x, y, label='Ideal', ls='--', color='gray')

    plt.ylabel('Run time (s)')
    plt.xlabel('Number of workers')
    plt.title('%s for %s' % (context, filename))
    plt.legend()
    plotfile = "%s_%s.png" % (filename.split('.')[0], context)
    print("    See plot file %s for scaling curves" % (plotfile))
    plt.savefig(plotfile)
    
    for label in ['dirty_max', 'dirty_min', 'psf_max', 'psf_min', 'restored_max',
                  'restored_min', 'deconvolved_max', 'deconvolved_min', 'residual_max', 'residual_min']:
        values = numpy.array([float(results[i][label]) for i, _ in enumerate(results)])
        values -= values[0]
        max_error = numpy.max(numpy.abs(values))
        if max_error > 1e-15:
            print("    Discrepancy between values for %s is %s:" % (label, max_error))
    
def read_results(filename, context='scaling'):
    results = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if row['context'] == context:
                results.append(row)
        csvfile.close()
    return results

if __name__ == '__main__':
    import csv
    
    print(sys.argv)
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            try:
                analzye_scaling(f)
            except KeyError:
                pass
            except IndexError:
                pass
            except ValueError:
                pass
