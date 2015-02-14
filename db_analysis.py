'''
db_stats.py : Extracts a few database's infos:
                - classes balance
                - images shapes
'''
import numpy as np
import tables as tb
import os
import numpy as np
import matplotlib.pylab as plt

# Parameters
db_path = '/home/cesar/DB/dogs_vs_cats'
db = os.path.join(db_path, 'train.h5')
TR_IDX = [0, 20000]
VA_IDX = [20000, 22500]
TE_IDX = [22500, 25000]

# Checking classes balance
#with tb.open_file(db, 'r') as f:
    #labels = [x[0] for x in f.root.Data.y.iterrows(TR_IDX[0], TR_IDX[1])]
    #balance = 100*(np.sum(labels)/float(len(labels)))
    #print('Classes balance : %.1f %% /  %.1f %%' %(balance, 100-balance))  

# Extracting size parameters
with tb.open_file(db, 'r') as f:
    shapes = [x for x in f.root.Data.s.iterrows(0, 20000)]
    

     
    xs = np.array([x[0] for x in shapes])
    ys = np.array([x[1] for x in shapes])
    plt.hexbin(xs, ys)
    plt.show()
