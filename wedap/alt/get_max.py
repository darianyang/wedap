#!/usr/bin/env python
# get_max.py
#
# Python script that goes through an h5 file and outputs the iteration and segment number
# with the maximum progress coordinate value
#
#

import numpy
import h5py
import sys

f = h5py.File("west.h5", 'r')

# phase 1: finding iteration number

max_values = []
for i in range(1,26): # change indices to the number of iterations you have
  i = str(i)
  iteration = "iter_" + str(numpy.char.zfill(i,8))
  pc = f['iterations'][iteration]['pcoord']
  maxv = numpy.max(pc[:,-1,0])  # change last digit for each progress coordinate
  max_values.append(maxv)
maxmax = numpy.max(max_values)
#print maxmax
nw = numpy.where(max_values>(0.99999*maxmax)) # change interval if giving 2+ iterations
if nw[0].shape[0] > 1:
  sys.exit("There are multiple maximums in nw.")
iter_num = str((nw[0]+1)[0])

# phase 2: finding seg number
wheretolook = "iter_" + str(numpy.char.zfill(iter_num,8))
max_iter = f['iterations'][wheretolook]['pcoord'][:,-1,0] # change last digit for each progress coordinate
segmax = numpy.max(max_iter)
nw2 = numpy.where(max_iter>(0.99999*segmax)) # change interval if giving 2+ iterations
if nw2[0].shape[0] > 1:
  sys.exit("There are multiple maximums in nw2.")
seg_num = (nw2[0])[0]
print( "go to iter " + str(iter_num) + ", " + "and seg " + str(seg_num))
