#!/usr/bin/python

# Copyright (c) 2020      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
# Simple example usage of SPC snapshots that creates a heatmap of
# SPC usage over time.

import sys
import glob
import operator
import struct

import numpy as np
import matplotlib
matplotlib.use('Agg') # For use with headless systems
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker

def combine(filename, data):
    f = open(filename, 'rb')
    for i in range(0,num_counters):
        temp = struct.unpack('l', f.read(8))[0]
        if 'TIME' in names[i]:
            temp /= freq_mhz
        data[i].append(temp)

def fmt(x, pos):
    return '{:,.0f}'.format(x)

# Make sure the proper number of arguments have been supplied
if len(sys.argv) < 4:
    print("Usage: ./spc_snapshot_parse.py [/path/to/data/files] [datafile_label] [list_of_spcs]")
    exit()

path = sys.argv[1]
label = sys.argv[2]

xml_filename = ''
# Lists for storing the snapshot data files from each rank
copies = []
ends = []
# Populate the lists with the appropriate data files
for filename in glob.glob(path + "/spc_data*"):
    if label in filename:
        if xml_filename == '' and '.xml' in filename:
            xml_filename = filename
        if '.xml' not in filename:
            temp = filename.split('/')[-1].split('.')
            if len(temp) < 5:
                temp[-1] = int(temp[-1])
                ends.append(temp)
            else:
                temp[-1] = int(temp[-1])
                temp[-2] = int(temp[-2])
                copies.append(temp)

# Sort the lists
ends = sorted(ends, key = operator.itemgetter(-1))
for i in range(0,len(ends)):
    ends[i][-1] = str(ends[i][-1])
copies = sorted(copies, key = operator.itemgetter(-2,-1))
for i in range(0,len(copies)):
    copies[i][-1] = str(copies[i][-1])
    copies[i][-2] = str(copies[i][-2])

sep = '.'

xml_file = open(xml_filename, 'r')
num_counters = 0
freq_mhz = 0
names = []
base = []
# Parse the XML file (same for all data files)
for line in xml_file:
    if 'num_counters' in line:
        num_counters = int(line.split('>')[1].split('<')[0])
    if 'freq_mhz' in line:
        freq_mhz = int(line.split('>')[1].split('<')[0])
    if '<name>' in line:
        names.append(line.split('>')[1].split('<')[0])
        value = [names[-1]]
        base.append(value)

prev = copies[0]
i = 0
ranks = []
values = []
times = []
time = []

# Populate the data lists
for n in range(0,len(base)):
    values.append([0, names[n]])
for c in copies:
    if c[-2] != prev[-2]:
        filename = path + "/" + sep.join(ends[i])
        combine(filename, values)

        ranks.append(values)
        times.append(time)
        for j in range(0, len(names)):
            temp = [ranks[0][j][0]]

        values = []
        time = []
        for n in range(0,len(base)):
            values.append([i+1, names[n]])
        i += 1

    filename = path + "/" + sep.join(c)
    time.append(int(filename.split('.')[-1]))
    combine(filename, values)
    prev = c

filename = path + "/" + sep.join(ends[i])
combine(filename, values)
ranks.append(values)
times.append(time)

spc_list = sys.argv[3].split(",")

for i in range(0, len(names)):
    fig = plt.figure(num=None, figsize=(7, 9), dpi=200, facecolor='w', edgecolor='k')

    plot = False
    # Only plot the SPCs of interest
    if names[i] in spc_list:
        plot = True

    map_data = []
    avg_x = []

    for j in range(0, len(ranks)):
        if avg_x == None:
            avg_x = np.zeros(len(times[j])-1)
        empty = True
        for k in range(2,len(ranks[j][i])):
            if ranks[j][i][k] != 0:
                empty = False
                break
        if not empty:
            if plot:
                xvals = []
                yvals = []
                for l in range(1, len(times[j])):
                    if ranks[j][i][l+2] - ranks[j][i][l+1] < 0:
                        break
                    xvals.append(times[j][l] - times[j][0])
                    yvals.append(ranks[j][i][l+2] - ranks[j][i][l+1])

                map_data.append(yvals)
                for v in range(0,len(avg_x)):
                    avg_x[v] += xvals[v]
    if plot:
        for v in range(0,len(avg_x)):
            avg_x[v] /= float(len(ranks))

        ax = plt.gca()
        im = ax.imshow(map_data, cmap='Reds', interpolation='nearest')

        cbar = ax.figure.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt))
        cbar.ax.set_ylabel("Counter Value", rotation=-90, va="bottom")

        plt.title(names[i] + ' Snapshot Difference')

        plt.xlabel('Time')
        plt.ylabel('MPI Rank')

        ax.set_xticks(np.arange(len(avg_x)))
        ax.set_yticks(np.arange(len(map_data)))
        ax.set_xticklabels(avg_x)

        plt.show()
        fig.savefig(names[i] + '.png')
