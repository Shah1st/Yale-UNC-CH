
import os
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from _cfg import cfg

import torch
import torch.nn.functional as F

def smart_preprocess_256x72(x):
    """
    Smart preprocessing: Early arrivals focus + minimal padding to 72
    
    Args:
        x: numpy array, shape (sources, time, receivers) - single sample
    
    Returns:
        numpy array, shape (sources, 256, 72)
    """
    x_tensor = torch.from_numpy(x).float()
    
   
    # This captures direct waves, first reflections, and early refractions
    time_samples = x_tensor.shape[1]
    crop_samples = min(512, time_samples)
    x_cropped = x_tensor[:, :crop_samples, :]  
    
    # Downsample time to 256 samples using area interpolation
    sources, _, receivers = x_cropped.shape
    x_reshaped = x_cropped.unsqueeze(1)  
    
    # Downsample to 256Г—70 first (preserve original spatial resolution)
    x_downsampled = F.interpolate(x_reshaped, size=(256, 70), mode='area')
    x_downsampled = x_downsampled.squeeze(1) 

    x_padded = F.pad(x_downsampled, (1, 1, 0, 0), mode='replicate') 
        
    return x_padded.numpy().astype(np.float16)

def inputs_files_to_output_files(input_files):
    """Convert input file paths to output file paths"""
    return [
        f.replace('/seis', '/vel').replace('/data', '/model')
        for f in input_files
    ]

def get_72x72_data_files(data_path):
    """Get data files for 72x72 mode"""
    # All filenames
    all_inputs = [
        f for f in glob.glob(data_path + "/*/*.npy")
        if ('/seis' in f) or ('/data' in f)
    ]
    all_outputs = inputs_files_to_output_files(all_inputs)
    assert all([x != y for x,y in zip(all_inputs, all_outputs)])

    # Validation filenames (same split as HGNet)
    val_fpaths= [
        'CurveFault_A/seis2_1_0.npy', 'CurveFault_A/seis2_1_1.npy', 
        'CurveFault_B/seis6_1_0.npy', 'CurveFault_B/seis6_1_1.npy', 
        'CurveVel_A/data1.npy', 'CurveVel_A/data10.npy', 
        'CurveVel_B/data1.npy', 'CurveVel_B/data10.npy', 
        'FlatFault_A/seis2_1_0.npy', 'FlatFault_A/seis2_1_1.npy', 
        'FlatFault_B/seis6_1_0.npy', 'FlatFault_B/seis6_1_1.npy', 
        'FlatVel_A/data1.npy', 'FlatVel_A/data10.npy', 
        'FlatVel_B/data1.npy', 'FlatVel_B/data10.npy', 
        'Style_A/data1.npy', 'Style_A/data10.npy', 
        'Style_B/data1.npy', 'Style_B/data10.npy',
    ]

    train_inputs, train_outputs = [], []
    valid_inputs, valid_outputs = [], []

    # Iterate and split files
    for a, b in zip(all_inputs, all_outputs):
        to_val = False
        
        for c in val_fpaths:
            if c in a:
                to_val = True

        if to_val:
            valid_inputs.append(a)
            valid_outputs.append(b)
        else:
            train_inputs.append(a)
            train_outputs.append(b)

    return train_inputs, train_outputs, valid_inputs, valid_outputs

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        cfg,
        mode = "train", 
    ):
        self.cfg = cfg
        self.mode = mode
        
        if cfg.use_72x72_mode:
            self.data, self.labels, self.records = self.load_72x72_metadata()
        else:
            self.data, self.labels, self.records = self.load_metadata()

    def load_metadata(self):
        """Original full resolution data loading"""
        # Select rows
        df= pd.read_csv("/kaggle/input/openfwi-preprocessed-72x72/folds.csv")
        
        if hasattr(self.cfg, 'subsample_fraction') and self.cfg.subsample_fraction is not None:
            # Sample by fraction within each dataset family
            df = df.groupby(["dataset", "fold"]).apply(
                lambda x: x.sample(frac=self.cfg.subsample_fraction, random_state=self.cfg.seed)
            ).reset_index(drop=True)
        elif self.cfg.subsample is not None:
            # Original fixed-number sampling
            df= df.groupby(["dataset", "fold"]).head(self.cfg.subsample)

        if self.mode == "train":
            df= df[df["fold"] != 0]
        else:
            df= df[df["fold"] == 0]

        data = []
        labels = []
        records = []
        mmap_mode = "r"

        for idx, row in tqdm(df.iterrows(), total=len(df), disable=self.cfg.local_rank != 0):
            row= row.to_dict()

           
            # Original full dataset paths
            p1 = os.path.join("/kaggle/input/open-wfi-1/openfwi_float16_1/", row["data_fpath"])
            p2 = os.path.join("/kaggle/input/open-wfi-1/openfwi_float16_1/", row["data_fpath"].split("/")[0], "*", row["data_fpath"].split("/")[-1])
            p3 = os.path.join("/kaggle/input/open-wfi-2/openfwi_float16_2/", row["data_fpath"])
            p4 = os.path.join("/kaggle/input/open-wfi-2/openfwi_float16_2/", row["data_fpath"].split("/")[0], "*", row["data_fpath"].split("/")[-1])
            farr = glob.glob(p1) + glob.glob(p2) + glob.glob(p3) + glob.glob(p4)
        
            # Map to lbl fpath
            farr= farr[0]
            flbl= farr.replace('seis', 'vel').replace('data', 'model')
            
            # Load
            arr= np.load(farr, mmap_mode=mmap_mode)
            lbl= np.load(flbl, mmap_mode=mmap_mode)

            # Append
            data.append(arr)
            labels.append(lbl)
            records.append(row["dataset"])

        return data, labels, records

    def load_72x72_metadata(self):
        """Load 72x72 preprocessed data"""
        train_inputs, train_outputs, valid_inputs, valid_outputs = get_72x72_data_files(self.cfg.data_path_72x72)
        
        if self.mode == "train":
            input_files = train_inputs
            output_files = train_outputs
        else:
            input_files = valid_inputs
            output_files = valid_outputs

        data = []
        labels = []
        records = []

        for input_file, output_file in tqdm(zip(input_files, output_files), total=len(input_files), disable=self.cfg.local_rank != 0):
            # Load data
            arr = np.load(input_file, mmap_mode='r')
            lbl = np.load(output_file, mmap_mode='r')
            
            # Extract dataset name from path
            dataset_name = input_file.split("/")[-2]
            
            data.append(arr)
            labels.append(lbl)
            records.append(dataset_name)

        return data, labels, records

    def __getitem__(self, idx):
        row_idx = idx // 500
        col_idx = idx % 500
    
        d = self.records[row_idx]
        x = self.data[row_idx][col_idx, ...]
        y = self.labels[row_idx][col_idx, ...]
    
        if self.mode == "train":
            # Flip augmentation (50% chance)
            if np.random.random() < 0.5:
                if self.cfg.use_72x72_mode:
                    x = x[::-1, :, ::-1]
                    y = y[:, ::-1]
                elif self.cfg.use_smart_256x72_mode:
                    x = x[::-1, :, ::-1]
                    y = y[:, ::-1]
                else:
                    x = x[::-1, :, ::-1]
                    y = y[..., ::-1]
                    
            if self.cfg.use_augmentations:
            
                # Configurable noise augmentation
                if np.random.random() < self.cfg.aug_noise_prob:
                    noise_level = self.cfg.aug_noise_level * np.std(x)
                    noise = np.random.normal(0, noise_level, x.shape).astype(x.dtype)
                    x = x + noise
                    
                    if np.isnan(x).any() or np.isinf(x).any():
                        print(f"Warning: NaN/inf detected in noise augmentation, reverting")
                        x = x - noise
                
                # Configurable scale augmentation
                if np.random.random() < self.cfg.aug_scale_prob:
                    scale = np.random.uniform(self.cfg.aug_scale_range[0], self.cfg.aug_scale_range[1])
                    x = x * scale
                    
                    if np.isnan(x).any() or np.isinf(x).any():
                        print(f"Warning: NaN/inf detected in scale augmentation, reverting")
                        x = x / scale
    
        x = x.copy()
        y = y.copy()
    
        if self.cfg.use_smart_256x72_mode:
            x = smart_preprocess_256x72(x)
        
        elif self.cfg.use_72x72_mode:
            pass
    
        if not self.cfg.mixed_precision:
            x = x.astype(np.float32)
            y = y.astype(np.float32)
        
        return x, y
    
    def __len__(self, ):
        return len(self.records) * 500