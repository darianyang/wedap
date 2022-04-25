"""
Check to make sure the aux datasets look okay.
"""

import h5py
import numpy as np
import pandas as pd

f = h5py.File("west_c2.h5", mode='r')

i = 1
string = "iterations/iter_" + str(i).zfill(8) + "/auxdata"

# parse h5 file to find auxillary dataset names
hdf_data = f[string]
auxlist = list(hdf_data)

for aux in auxlist:
    df = pd.DataFrame(f[string + "/" + aux])
    print(f"{aux}:\n{df.head(10)}\n")

