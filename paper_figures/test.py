import wedap
import matplotlib.pyplot as plt

# if there is alot of plotting choices, it can be easier to just put it all into a dictionary
plot_options = {"h5" : "west.h5",
                "data_type" : "average", 
                "plot_mode" : "contour", 
                "Xname" : "pcoord", 
                "Yname" : "RoG", 
                "zlabel" : "-RT ln(P) (kcal/mol)",
                "p_units" : "kcal",
                "xlabel" : "Heavy Atom RMSD ($\AA$)",
                "ylabel" : "Radius of Gyration ($\AA$)",
                "contour_interval" : 0.2,
                "proj3d" : True,
                }
wedap.H5_Plot(**plot_options).plot()
plt.tight_layout()
plt.show()
