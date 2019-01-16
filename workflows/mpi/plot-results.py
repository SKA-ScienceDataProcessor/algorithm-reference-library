#!/usr/bim/python

# Script to plot the performance/scaling results

import os
import sys

sys.path.append(os.path.join('..', '..'))


results_dir = './results/timing-csd3/16'
#from matplotlib import pylab

#pylab.rcParams['figure.figsize'] = (12.0, 12.0)
#pylab.rcParams['image.cmap'] = 'rainbow'

import numpy

from matplotlib import pyplot as plt


def read_results_file(filename):
    """ Read the results from a file and returns them as structured numpy array
    The order is (number_nodes, number_procs, numfreqw, time in sec)
    :param filename: filename
    :return: List of tuples as above
    """
    d=numpy.loadtxt('%s/%s' %(results_dir,filename),
                    dtype={'names': ('numnodes','numprocs','nfreqw','time'),
                           'formats': ('i','i','i','f')},
                    delimiter='\t')
    print(d)
    return d

def plot_freqwin(data,fignum,figtitle):
    """ data is a 1d numpy array with the structure:
        names': ('numnodes','numprocs','nfreqw','time'),
        formats': ('i','i','i','f')},
    """
    nfreqwin_list=[71,101,203,407]
    plot_shapes=['ro','bs','g^','y*']
    fig=plt.figure(fignum)
    fig.suptitle(figtitle)
    for i,nfreqwin in enumerate(nfreqwin_list): 
        procs= [x['numprocs'] for x in data if x['nfreqw']==nfreqwin]
        nodes= [x['numnodes'] for x in data if x['nfreqw']==nfreqwin]
        times= [x['time'] for x in data if x['nfreqw']==nfreqwin]
        plt.plot(procs,times,plot_shapes[i],linestyle='--',label="nfreqwin %d" %
             (nfreqwin))
        plt.ylabel('time in seconds')
        plt.xlabel('number of processes')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(prop={'size':10})
    #fig.savefig('%s-fig.jpg'%(figtitle))
    return
        


def main():
    file_list=['predict','invert','contimg','ical']
    for i,f in enumerate(file_list):

        d=read_results_file("%s-results.txt"%(f))
        plot_freqwin(d,i,f)
    plt.show()
    return


if __name__ == '__main__':
    main()
