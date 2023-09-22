import mdap
import numpy as np

data_path = "mdap/tests/data/"

# # timeseries
# for Xname in ["input_data_0.dat", "input_data_2.npy", "input_data_2.pkl"]:
#     X, Y, _ = mdap.MD_Pdist(Xname=data_path + Xname, data_type="time").pdist()
#     np.savetxt(f"mdap/tests/data/timeseries_{Xname}_X.txt", X)
#     np.savetxt(f"mdap/tests/data/timeseries_{Xname}_Y.txt", Y)

# # 1D pdists
# for Xname in ["input_data_3.dat"]:
#     for Xindex in [1, 2]:
#         for Xinterval in [1, 10]:
#             X, Y, _ = mdap.MD_Pdist(data_type="pdist", Xname=data_path + Xname, 
#                                     Xindex=Xindex, Xinterval=Xinterval).pdist()
#             np.savetxt(f"mdap/tests/data/pdist1d_{Xname}_idx{Xindex}_int{Xinterval}_X.txt", X)
#             np.savetxt(f"mdap/tests/data/pdist1d_{Xname}_idx{Xindex}_int{Xinterval}_Y.txt", Y)

# # 2D pdists
# for Xname in ["input_data_0.dat", "input_data_2.npy", "input_data_2.pkl"]:
#     for Yname in ["input_data_3.dat"]:
#         for Yindex in [1, 2]:
#             X, Y, Z = mdap.MD_Pdist(data_type="pdist", Xname=data_path + Xname, Xinterval=10,
#                                     Yname=data_path + Yname, Yindex=Yindex).pdist()
#             np.savetxt(f"mdap/tests/data/pdist2d_{Xname}_{Yname}_yidx{Yindex}_X.txt", X)
#             np.savetxt(f"mdap/tests/data/pdist2d_{Xname}_{Yname}_yidx{Yindex}_Y.txt", Y)
#             np.savetxt(f"mdap/tests/data/pdist2d_{Xname}_{Yname}_yidx{Yindex}_Z.txt", Z)
        
# 3D scatter
for Xname in ["input_data_0.dat"]:
    for Yname in ["input_data_1.dat", "input_data_2.npy"]:
        for Zname in ["input_data_2.dat", "input_data_2.pkl"]:
            X, Y, Z = mdap.MD_Pdist(data_type="pdist", 
                                    Xname=data_path + Xname, 
                                    Yname=data_path + Yname, 
                                    Zname=data_path + Zname).pdist()
            np.savetxt(f"mdap/tests/data/pdist3d_{Xname}_{Yname}_{Zname}_X.txt", X)
            np.savetxt(f"mdap/tests/data/pdist3d_{Xname}_{Yname}_{Zname}_Y.txt", Y)
            np.savetxt(f"mdap/tests/data/pdist3d_{Xname}_{Yname}_{Zname}_Z.txt", Z)