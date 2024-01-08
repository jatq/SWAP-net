import pandas as pd
import h5py

import numpy as np

def get_data_azimuths(root, split = 'train'):
    meta = pd.read_csv(root+'/meta.csv', sep='\t', low_memory=False)
    if split == 'train':
        meta = meta[meta.is_train]
    else:
        meta = meta[~meta.is_train]
    azimuths = meta.azimuth.to_numpy()

    data = []
    with h5py.File(root+'/data.hdf5', 'r') as h5:
        for name in meta.trace_name:
            data.append(
                np.array(h5.get(f'data/{name}'))
            )
    data = np.array(data).astype(np.float32)
    return data, azimuths


