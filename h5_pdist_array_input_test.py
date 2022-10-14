import wedap
import numpy as np
import matplotlib.pyplot as plt

#total_array_out = np.loadtxt("p53_X_array.txt")
total_array_out = np.loadtxt("p53_X_array_i15.txt")
# original_array = np.loadtxt("p53_X_array_noreshape.txt")

def test_func(data):
    #return data[:,::10,:]    
    return data + 10

#h5 = wedap.H5_Plot(h5="data/p53.h5", data_type="evolution")
#print(h5.reshape_total_data_array(total_array_out).shape)
#print(total_array_out[::1].shape)

# test of p53 output data as input
# note that this dataset is not correct since I took it from the truncated p53.h5 file which shouldn't have 
# the last /west_current_iter filled out but does
#pdist = wedap.H5_Pdist("data/p53.h5", "evolution", data_proc=lambda data : data[:,::10,:])
#pdist = wedap.H5_Pdist("data/p53.h5", "evolution", Xname=, data_proc=lambda data : data[:,::10,:])
# pdist = wedap.H5_Pdist("data/p53.h5", "evolution", Xname=total_array_out, data_proc=test_func, 
#                        H5save_out="Xsave_p53_test_15i.h5", Xsave_name="pcoord_plus_ten", last_iter=15)
#pdist = wedap.H5_Pdist("Xsave_p53_test.h5", "evolution", Xname="pcoord_plus_ten")

# pdist = wedap.H5_Pdist("data/p53.h5", "evolution", last_iter=15, Xname=total_array_out)
# X, Y, Z = pdist.pdist()
# wedap.H5_Plot(X, Y, Z).plot()
# plt.show()

# now instead of testing p53.h5, using a real rmsd based 2kod v02 we dataset
o_angle = np.loadtxt("data/2kod_rms_v02_o_angle_10i.dat")[:,1]
rog = np.loadtxt("data/2kod_rms_v02_rog_10i.dat")[:,1]
# since the dataset being used is 10i, need to also do this to h5 data, hence the data_proc arg
#pdist = wedap.H5_Pdist("data/2kod_rms_we_v02.h5", "evolution", Xname=o_angle, data_proc=lambda data : data[:,::10,:])
# this should have an error (need to make both input and h5 data 10 intervaled)
#pdist = wedap.H5_Pdist("data/2kod_rms_we_v02.h5", "evolution", Xname=o_angle)
#X, Y, Z = pdist.pdist()
#print(X.shape)
# wedap.H5_Plot(h5="data/2kod_rms_we_v02.h5", data_type="evolution", 
#               Xname=o_angle, data_proc=lambda data : data[:,::10,:]).plot()
# wedap.H5_Plot(h5="data/2kod_rms_we_v02.h5", data_type="evolution", histrange_x=[16,20],
#               Xname="RoG", data_proc=lambda data : data[:,::10,:]).plot()
# wedap.H5_Plot(h5="data/2kod_rms_we_v02.h5", data_type="evolution", histrange_x=[16,20],
#               Xname=rog, data_proc=lambda data : data[:,::10,:]).plot()

# average and instant
wedap.H5_Plot(h5="data/2kod_rms_we_v02.h5", data_type="average", 
              Yname=o_angle, data_proc=lambda data : data[:,::10,:]).plot()

plt.show()


# TODO: allow user to load in a 1d array of data, then use this to make pdist
# maybe have some way to allow for intervaled input data to be compatible
# so if user ran analysis every 10 frames, wedap can still work with it
# could add this to the reshape method functionality
#   Or! could include this in the data proc/func feature for XYZ
#   Note that I could store the function in an array and make it so if
#   there is one item, use it for all dim, other wise if items = dims
#   use each accordingly? Then only will have one func/proc variable 


###### clean example

# import wedap
# import numpy as np
# import matplotlib.pyplot as plt

# # load dataset and get the 1d array of data
# o_angle = np.loadtxt("data/2kod_rms_v02_o_angle_10i.dat")[:,1]

# # since the dataset being used is 10i, need to also do this to h5 data, hence the data_proc arg
# # normally, Xname would be a string e.g. "pcoord" or "rmsd_bb" (aux name)
# wedap.H5_Plot(h5="data/2kod_rms_we_v02.h5", data_type="evolution", 
#               Xname=o_angle, data_proc=lambda data : data[:,::10,:]).plot()

# plt.show()