# An example script to sort Gaussians from a .ply file with PLAS.

# The input is a .ply file with the Gaussians, and the output is a .ply file with the sorted Gaussians.
# For sorting, it is only using the 3D coordinates and the RGB colors (SH DC component) of the Gaussians.

# Note that sorting a .ply after training the model is much less efficient than sorting the Gaussians during training,
# and applying a regularization on the sorted grid. See results int Table 3 of the paper (https://arxiv.org/abs/2312.13299).

import numpy as np
import pandas as pd
import torch
from plyfile import PlyData, PlyElement
import trimesh as tm
import click
import os

import math
import re
from PIL import Image
import matplotlib.pyplot as plt
from .plas_playground.core import sort_with_plas
from .plas_playground.vad import compute_vad
from .utils import custom_quantize, custom_dequantize 
import time

# process fewer elements for development testing
# DEBUG_TRUNCATE_ELEMENTS = 1_000_000
DEBUG_TRUNCATE_ELEMENTS = None

COORDS_SCALE = 255
RGB_SCALE = 255

C0 = 0.28209479177387814
MIN_BLOC_SIZE=16

def prune_gaussians(df, num_to_keep):
    """Very crude pruning method that uses scaling and opacity to determine the impact of a Gaussian splat.
       We need this method to drop a few Gaussians to make them fit a square image.

       For a more sophisticated method, see e.g. "LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS"
       https://arxiv.org/abs/2311.17245
    """

    # from gaussian model:
    # self.scaling_activation = torch.exp
    # self.scaling_inverse_activation = torch.log

    # self.opacity_activation = torch.sigmoid
    # self.inverse_opacity_activation = inverse_sigmoid

    scaling_act = np.exp
    opacity_act = lambda x: 1 / (1 + np.exp(-x))

    # does this perhaps remove too many small points in the center of the scene, that form a bigger object?
    df["impact"] = scaling_act((df["s0"] + df["s1"] + df["s2"]).astype(np.float64)) * opacity_act(df["alpha"].astype(np.float64))

    df = df.sort_values("impact", ascending=False)
    df = df.head(num_to_keep)

    return df

def SH2RGB(sh):
    return sh * C0 + 0.5


def pre_process_df(df, sidelen, device):
    '''
    This function extracts the relevant fields from the dataframe, converts them to torch tensors ready to be sorted (right shape, size and device)
    '''
    # 1) # 2) ----------- extract fields as numpy -----------
    # scale colors to [0, RGB_SCALE] for sorting (original values are kept)
    x = df.loc[:, df.columns.str.startswith("x_")].values
    y = df.loc[:, df.columns.str.startswith("y_")].values
    z = df.loc[:, df.columns.str.startswith("z_")].values
    ##
    ##    x_quantized, r_min, r_max = custom_quantize(x)
    ##    y_quantized, r_min, r_max = custom_quantize(y)
    ##    z_quantized, r_min, r_max = custom_quantize(z)
    ##    
    r = df.loc[:, df.columns.str.startswith("r_")].values
    g = df.loc[:, df.columns.str.startswith("g_")].values
    b = df.loc[:, df.columns.str.startswith("b_")].values
    
    ##    
    q0 = df.loc[:, df.columns.str.startswith("q0_")].values
    q1 = df.loc[:, df.columns.str.startswith("q1_")].values
    q2 = df.loc[:, df.columns.str.startswith("q2_")].values
    q3 = df.loc[:, df.columns.str.startswith("q3_")].values
    ##    
    ##    q0_quantized, r_min, r_max = custom_quantize(q0)
    ##    q1_quantized, r_min, r_max = custom_quantize(q1)
    ##    q2_quantized, r_min, r_max = custom_quantize(q2)
    ##    
    s0 = df.loc[:, df.columns.str.startswith("s0")].values
    s1 = df.loc[:, df.columns.str.startswith("s1")].values
    s2 = df.loc[:, df.columns.str.startswith("s2")].values
    ##    
    alpha = df.loc[:, df.columns.str.startswith("alpha")].values
    print([df.columns.str.startswith("alpha")])
    print("alpha shape ", alpha.shape)
    # 2) ----------- convert to torch -----------
    x_torch, y_torch, z_torch = torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device), torch.from_numpy(z).float().to(device)
    q0_torch, q1_torch, q2_torch = torch.from_numpy(q0).float().to(device), torch.from_numpy(q1).float().to(device), torch.from_numpy(q2).float().to(device)
    s0_torch, s1_torch, s2_torch = torch.from_numpy(s0).float().to(device), torch.from_numpy(s1).float().to(device), torch.from_numpy(s2).float().to(device)
    alpha_torch = torch.from_numpy(alpha).float().to(device)
    print("alpha_torch shape", alpha_torch.shape)

    ##trajectory vectors x
    #x_traj_vals = df.loc[:, df.columns.str.startswith("x")].values
    #x_traj_torch = torch.from_numpy(x_traj_vals).float().to(device)
    ##trajectory vectors y
    #y_traj_vals = df.loc[:, df.columns.str.startswith("y")].values
    #y_traj_torch = torch.from_numpy(y_traj_vals).float().to(device)
    ##trajectory vectors x
    #z_traj_vals = df.loc[:, df.columns.str.startswith("z")].values
    #z_traj_torch = torch.from_numpy(z_traj_vals).float().to(device)
    # params to sort: 6D (3D coords + 3D colors + trajectory vectors)
    #print("x_traj_torch shape", x_traj_torch.shape)
    #print("y_traj_torch shape", y_traj_torch.shape)
    #print("z_traj_torch shape", z_traj_torch.shape)
    
    #only keep the 10 first trajectory vectors for now, to reduce the dimensionality for sorting and see if it already gives good results (we can keep more traj vectors later if needed)
    x_traj_torch = x_torch[:, :1]
    y_traj_torch = y_torch[:, :1]
    z_traj_torch = z_torch[:, :1]
    r_traj_torch = 100*r[:, :60]
    g_traj_torch = 100*g[:, :60]
    b_traj_torch = 100*b[:, :60]
    s0 = torch.from_numpy(df["s0"].values.reshape(-1, 1)).float().to(device)
    s1 = torch.from_numpy(df["s1"].values.reshape(-1, 1)).float().to(device)
    s2 = torch.from_numpy(df["s2"].values.reshape(-1, 1)).float().to(device)
    q0 = torch.from_numpy(df["q0_0"].values.reshape(-1, 1)).float().to(device)
    q1 = torch.from_numpy(df["q1_0"].values.reshape(-1, 1)).float().to(device)
    q2 = torch.from_numpy(df["q2_0"].values.reshape(-1, 1)).float().to(device)
    q3 = torch.from_numpy(df["q3_0"].values.reshape(-1, 1)).float().to(device)
    dc_vals = np.concatenate([r_traj_torch, g_traj_torch, b_traj_torch], axis=1)
    rgb_torch = torch.from_numpy(dc_vals).float().to(device)
    print("s0 shape", s0.shape)
    print("s1 shape", s1.shape)
    print("s2 shape", s2.shape)
    print("q1 shape", q1.shape)
    print("q2 shape", q2.shape)
    print("q3 shape", q3.shape)
    params = torch.cat([x_traj_torch, y_traj_torch, z_traj_torch, rgb_torch, alpha_torch, s0, s1, s2, q0, q1, q2, q3], dim=1) #rgb_torch], dim=1) #, 
    #params2 = torch.cat([rgb_torch], dim=1) #rgb_torch], dim=1) #, 
    #params = torch.cat([params1, params2, alpha_torch], dim=1) # (N, C)
    print("SORTING ACCORDING TO XYZ ONLY FOR NOW, params shape : ", params.shape)

    params_torch_grid = params.permute(1, 0).reshape(-1, sidelen, sidelen) # permute => channels first (C, H, W)
    
    return params_torch_grid
    # 32 GB in ~8mn

def sort_dyn_gaussians(df, resume_from_last = False, init_order= None, seq="Base", exp="Base"):
    #print("sorted df is df")
    #print("df comumns before pruning: ", df.columns)
    #df = prune_gaussians(df, int(np.sqrt(len(df)))**2)
    #return df
    t0=time.time()
    
    if (resume_from_last and os.path.exists("./sorted_indices.npy")):
        print("[sort_dyn_3d_gaussians.py] Pre-computed sorted indexes stored at ./sorted_indices.npy found, resuming from last ")
        sorted_indices = np.load("./sorted_indices.npy")
        num_gaussians = len(df)
        sidelen = int(np.sqrt(num_gaussians))
        df = prune_gaussians(df, sidelen * sidelen)
        orig_vad = compute_vad(df.values.reshape(sidelen, sidelen, -1))
        print(f"VAD of ply: {orig_vad:.4f}")
    
    else :
        print("[sort_dyn_3d_gaussians.py] Re-computing sorted indexes from scratch ")
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available(): 
            print("Sorting on cuda GPU")
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Running on MAC, you payed too much for your computer and it's still slow -- not tested")
            device = torch.device("mps") 
        else:
            print("Running on CPU, this may be slow -- consider using a machine with a CUDA-capable GPU -- not tested")
            device = "cpu"

        print(f"Using device: {device}")
        num_gaussians = len(df)
        sidelen = int(np.sqrt(num_gaussians))
        df = prune_gaussians(df, sidelen * sidelen)
        orig_vad = compute_vad(df.values.reshape(sidelen, sidelen, -1))
        print(f"VAD of ply: {orig_vad:.4f}")
        
        
        ## ----- 1) Print all column names
        #print("DataFrame columns:")
        #for col in df.columns:
        #    print(col)

        ## ----- 2) Save bar plots/ histograms of value distributions for each column
        #output_dir = "./temp"
        #os.makedirs(output_dir, exist_ok=True)
        #print("\nSaving distribution plots...")
        #for col in df.columns:
        #    values = df[col].dropna().values
        #    plt.figure(figsize=(6, 4))
        #    plt.hist(values, bins=200, edgecolor='black')
        #    plt.title(f"Distribution of {col}")
        #    plt.xlabel("Value")
        #    plt.ylabel("Frequency")
        #    save_path = os.path.join(output_dir, f"{col}.png")
        #    plt.savefig(save_path, dpi=150, bbox_inches='tight')
        #    plt.close()
        #    print(f"Saved: {save_path}")
        #    print("\nAll plots saved in ./temp/")

        # shuffling the input to avoid getting stuck in a local minimum with the sorting
        
        df = df.sample(frac=1)
        shuffled_vad = compute_vad(df.values.reshape(sidelen, sidelen, -1))
        print(f"VAD of shuffled ply: {shuffled_vad:.4f}")
        
        params_torch_grid = pre_process_df(df, sidelen, device)

        print("before sort_with_plas -- params_torch_grid.shape : ", params_torch_grid.shape )
        sorted_coords, sorted_grid_indices = sort_with_plas(params_torch_grid, MIN_BLOC_SIZE, improvement_break=1e-4, verbose=True)


        sorted_indices = sorted_grid_indices.flatten().cpu().numpy()

        # Save to .npy file
        np.save("./sorted_indices.npy", sorted_indices)
        print("the sorted_indices.npy file should just have been saved", flush=True)

    #remove all attributes that are coded in float64
    sorted_df = df.iloc[sorted_indices]
    
    for col in sorted_df.columns:
        if sorted_df[col].dtype==np.float64:
            print("dropping col ", col)
            sorted_df = sorted_df.drop(columns=[col])

    sorted_vad = compute_vad(sorted_df.values.reshape(sidelen, sidelen, -1))
    
    print(f"VAD of sorted ply: {sorted_vad:.4f}")
    t1 = time.time()
    print(f"Sorting completed in {t1 - t0:.2f} seconds.")
    return sorted_df

if __name__ == "__main__":
    sort_dyn_gaussians("/home/erikmamet/Current_proj/Dynamic3DGaussians/playground/compressed_outputs/compression_com_decomp_test_5_basketball_20260128_075454")

