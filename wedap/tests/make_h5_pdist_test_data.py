import wedap
import numpy as np

h5 = "wedap/data/p53.h5"

# for Xname in ["pcoord"]:
#     for Xindex in [0, 1]:
#         evolution = wedap.H5_Pdist(h5=h5, data_type="evolution", Xname=Xname, Xindex=Xindex)
#         X, Y, Z = evolution.pdist()

#         # X data is the variably filled array of instance pdist x values
#         np.savetxt(f"wedap/tests/data/evolution_{Xname}{Xindex}_X.txt", X)

#         # Z data is the pdist values of each iteration
#         np.savetxt(f"wedap/tests/data/evolution_{Xname}{Xindex}_Z.txt", Z)

# for Xname in ["pcoord"]:
#     for Yname in ["dihedral_3", "pcoord"]:
#         for Xindex in [0, 1]:
#             X, Y, Z = wedap.H5_Pdist(h5=h5, data_type="instant", Xindex=Xindex,
#                                      Xname=Xname, Yname=Yname).pdist()
#             np.savetxt(f"wedap/tests/data/instant_{Xname}{Xindex}_{Yname}_X.txt", X)
#             np.savetxt(f"wedap/tests/data/instant_{Xname}{Xindex}_{Yname}_Y.txt", Y)
#             np.savetxt(f"wedap/tests/data/instant_{Xname}{Xindex}_{Yname}_Z.txt", Z)

for Xname in ["dihedral_3", "pcoord"]:
    for Yname in ["pcoord"]:
        for Yindex in [0, 1]:
            X, Y, Z = wedap.H5_Pdist(h5=h5, data_type="average", Yindex=Yindex,
                                     Xname=Xname, Yname=Yname).pdist()
            np.savetxt(f"wedap/tests/data/average_{Xname}_{Yname}{Yindex}_X.txt", X)
            np.savetxt(f"wedap/tests/data/average_{Xname}_{Yname}{Yindex}_Y.txt", Y)
            np.savetxt(f"wedap/tests/data/average_{Xname}_{Yname}{Yindex}_Z.txt", Z)
