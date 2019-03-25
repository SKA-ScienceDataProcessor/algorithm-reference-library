import sys
import csv
import numpy

import matplotlib.pyplot as plt

def plot_xy(td, x='nfreqwin', y='time ICAL', label_key='rmax', title=None,
            csvfile='csvfile.csv', plot_type='plot'):
    plt.clf()
    
    xval = numpy.array(td[x]).astype('float')
    yval = numpy.array(td[y]).astype('float')
    labels = numpy.array(numpy.unique(td[label_key])).astype('str')
    xmax = 0.0
    ymax = 0.0
    for label in labels:
        xsel = list()
        ysel = list()
        for col in range(len(xval)):
            if td[label_key][col] == label:
                xsel.append(xval[col])
                ysel.append(yval[col])
        xsel = numpy.array(xsel)
        ysel = numpy.array(ysel)
        order = numpy.argsort(xsel)
        ysel = ysel[order]
        xsel = xsel[order]
        if plot_type == 'semilogx':
            plt.semilogy(xsel, ysel, '-', label=label)
            plt.semilogy(xsel, ysel, '-', label=label)
        elif plot_type == 'semilogy':
            plt.semilogy(xsel, ysel, '-', label=label)
        else:
            xmax = max(xmax, numpy.max(xsel))
            ymax = max(ymax, numpy.max(ysel))
            plt.plot(xsel, ysel, '-', label=label)
    plt.ylim(0.0, 1.1 * ymax)
    plt.xlim(0.0, 1.1 * xmax)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    csvfile_trimmed = csvfile.split('/')[0]
    if title is not None:
        plt.title(title)
    else:
        plt.title("%s: %s vs %s, labs %s" % (csvfile_trimmed, y, x, label_key))
    filename = csvfile.replace('.csv', '')
    filename = "%s_%s_%s_%s.jpg" % (filename, x, y, label_key)
    plt.grid(True, 'both')
    plt.savefig(filename)
    plt.show()

csvfiles = ["scaling_300_dist/scaling_300_dist.csv",
            "scaling_300_ser/scaling_300_ser.csv",
            "scaling_600_dist/scaling_600_dist.csv",
            "scaling_600_ser/scaling_600_ser.csv",
            "scaling_600_ser_f512/scaling_600_ser_f512.csv"]

csvfiles = ["scaling_1200_all/scaling_1200_all.csv"
            ]
import glob

csvfiles = glob.glob('mftests_*/*.csv')
print(csvfiles)

for csvfile in csvfiles:
    f = open(csvfile, 'r')
    td={}
    dictreader = csv.DictReader(f)
    for d in dictreader:
        for key in d.keys():
            if key not in td.keys():
                td[key] = list()
            td[key].append(d[key])

    plot_type='plot'
    plot_xy(td, x='nfreqwin', y='time ICAL graph', csvfile=csvfile, label_key='nworkers', plot_type=plot_type)
    plot_xy(td, x='size ICAL graph', y='time ICAL graph', csvfile=csvfile, label_key='nworkers', plot_type=plot_type)

    plot_xy(td, x='nfreqwin', y='time ICAL', csvfile=csvfile, label_key='nworkers', plot_type=plot_type)
    plot_xy(td, x='nworkers', y='time ICAL', csvfile=csvfile, label_key='nfreqwin', plot_type=plot_type)

    plot_xy(td, x='nfreqwin', y='time invert', csvfile=csvfile, label_key='nworkers', plot_type=plot_type)
    plot_xy(td, x='nworkers', y='time invert', csvfile=csvfile, label_key='nfreqwin', plot_type=plot_type)

    plot_xy(td, x='nworkers', y='time overall', csvfile=csvfile, label_key='nfreqwin', plot_type=plot_type)

