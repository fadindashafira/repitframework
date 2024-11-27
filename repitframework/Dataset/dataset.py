from torch.utils.data import Dataset
from repitframework import config, OpenFOAM
import numpy as np
import pandas as pd
from pathlib import Path

base_config = config.BaseConfig()

class MLPDataset(Dataset):
    def __init__(self, data, target):
        super(MLPDataset, self).__init__(data, target)
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

class FVMNDataset(Dataset):
    def __init__(self, data_path:Path, start_time, end_time,zero_padding:bool=True):
        super(FVMNDataset, self).__init__(data_path)
        self.data_path = data_path
        self.start_time = start_time
        self.end_time = end_time
        self.vars: list = base_config.data_vars
        self.time_step = base_config.time_step    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def parse_numpy(self, data_path:Path):
        data:np.ndarray = np.load(data_path)
        if base_config.data_dim == 2:
            assert data.shape[0] == base_config.grid_x * base_config.grid_y, "check data shape and grid size mentioned in config."
            # Why order="F"? Check: https://github.com/JBNU-NINE/repit_container/blob/main/repit_wiki/Data-Loader-for-FVMN.md
            return data.reshape(base_config.grid_x, base_config.grid_y, order="F")
        elif base_config.data_dim == 3:
            assert data.shape[0] == base_config.grid_x * base_config.grid_y * base_config.grid_z, "check data shape and grid size mentioned in config."
            return data.reshape(base_config.grid_x, base_config.grid_y, base_config.grid_z, order="F")
        else:
            return data
    
    def add_zero_padding(self, data:np.ndarray):
        return np.pad(data, 1, mode="constant", constant_values=0)
        
    def add_feature(self, padded_matrix:np.ndarray):
        window_shape = (3, 3)
        sliding_window = np.lib.stride_tricks.sliding_window_view(padded_matrix, window_shape)
        x,y = window_shape[0] // 2, window_shape[1] // 2 
        correlated_features = np.stack([
            sliding_window[:,:,x,y],
            sliding_window[:,:,x-1,y],
            sliding_window[:,:,x+1,y],
            sliding_window[:,:,x,y-1],
            sliding_window[:,:,x,y+1]
        ], axis=-1)
        return correlated_features.reshape(-1, 5)
    
    def prepare_input(self):
        pass

    def calculate_difference(self):
        pass
    
    def normalize_input(self):
        pass

    def test_train_split(self):
        pass 

class CNNDataset(Dataset):
    def __init__(self, data, target):
        super(CNNDataset, self).__init__(data, target)

class RNNDataset(Dataset):
    def __init__(self, data, target):
        super(RNNDataset, self).__init__(data, target)