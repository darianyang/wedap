import wedap
import numpy as np
import matplotlib.pyplot as plt

total_array_out = np.loadtxt("p53_X_array.txt")
original_array = np.loadtxt("p53_X_array_noreshape.txt")

def test_func(data):
    """
    Take every 10 frames.
    """
    #return data[:,::10,:]    
    return data + 10

#h5 = wedap.H5_Plot(h5="data/p53.h5", data_type="evolution")
#print(h5.reshape_total_data_array(total_array_out).shape)

print(total_array_out[::1].shape)

#pdist = wedap.H5_Pdist("data/p53.h5", "evolution", data_proc=lambda data : data[:,::10,:])
pdist = wedap.H5_Pdist("data/p53.h5", "evolution", Xname=total_array_out, data_proc=test_func)
X, Y, Z = pdist.pdist()
#print(X.shape)
wedap.H5_Plot(X, Y, Z).plot()
plt.show()


# TODO: allow user to load in a 1d array of data, then use this to make pdist
# maybe have some way to allow for intervaled input data to be compatible
# so if user ran analysis every 10 frames, wedap can still work with it
# could add this to the reshape method functionality
#   Or! could include this in the data proc/func feature for XYZ
#   Note that I could store the function in an array and make it so if
#   there is one item, use it for all dim, other wise if items = dims
#   use each accordingly? Then only will have one func/proc variable 