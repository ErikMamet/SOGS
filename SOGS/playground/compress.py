import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json 
import time
import ffmpeg 
import json
import torch 

from .extract_pretrained_dyn_3dgs import saved_npz_to_df, df_to_params
from .sort_dyn_3d_gaussians import prune_gaussians

# Determine the path to the SOGS folder relative to this script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SOGS_DIR = os.path.join(THIS_DIR, "SOGS")

# Add SOGS to sys.path so its submodules can be imported as top-level modules
if SOGS_DIR not in sys.path:
    sys.path.insert(0, SOGS_DIR)

# Optional: verify paths
#print("sys.path:", sys.path)

### Get the root directory dynamically
##root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
##sys.path.append(root_dir)
##print("sys path ", sys.path)

# absolute imports must come after adding REPO_ROOT
from ..compression import compression_exp as compr
from .sort_dyn_3d_gaussians import sort_dyn_gaussians, prune_gaussians
from .extract_pretrained_dyn_3dgs import saved_npz_to_df


# some helper functions to compress and decompress dataframes using compression_exp.py

def compress_df(df, out_folder_path):
    print("starting compression")
    os.makedirs(out_folder_path, exist_ok=True)
    attr_configs = []
    for column in df.columns:
        attr_configs.append({
            'name': column,
            'method': 'jpeg-xl'
        })
        
        save_path, out, out = compr.compress_attr(attr_configs[-1], df, out_folder_path)

def decompress_df(reconstructed_df_dir):
    reconstructed_df = pd.DataFrame()
    attr_configs = []
    names = [name.split(".")[0] for name in os.listdir(reconstructed_df_dir) if name.endswith(".jxl")]
    for column in names:
        attr_configs.append({
            'name': column,
            'method': 'jpeg-xl'
        })
        
        decompressed_attribute = compr.decompress_attr(attr_configs[-1], os.path.join(reconstructed_df_dir, column+".jxl"), None, None)
        reconstructed_df[column] = decompressed_attribute.flatten()

    return reconstructed_df


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, json_path: str = None):
    report = {}

    # ---- 1) Compare column names ----
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    report["columns_only_in_df1"] = sorted(list(cols1 - cols2))
    report["columns_only_in_df2"] = sorted(list(cols2 - cols1))
    report["common_columns"] = sorted(list(cols1 & cols2))

    # ---- 2) Compare column dtypes ----
    dtype_mismatches = {}
    dtype_matches = []
    for col in report["common_columns"]:
        if df1[col].dtype != df2[col].dtype:
            dtype_mismatches[col] = {
                "df1_dtype": str(df1[col].dtype),
                "df2_dtype": str(df2[col].dtype)
            }
        else:
            dtype_matches.append(col)

    report["dtype_mismatches"] = dtype_mismatches
    report["dtype_matches"] = dtype_matches

    # ---- 3) Compute MSE for numeric matching columns ----
    mse = {}
    for col in dtype_matches:
        if pd.api.types.is_numeric_dtype(df1[col]):
            diff = df1[col].to_numpy() - df2[col].to_numpy()
            mse[col] = float(np.mean(diff ** 2))

    report["mse"] = mse

    # ---- 4) Save to JSON if path is provided ----
    if json_path is not None:
        with open(json_path, "w") as f:
            json.dump(report, f, indent=4)

    return report

def get_latest_output_dir(exp, seq):
    base_output_dir = './playground/compressed_outputs'
    dirs = [d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d))]
    filtered_dirs = [d for d in dirs if d.startswith(f'compression_{exp}_{seq}_')]
    if not filtered_dirs:
        raise FileNotFoundError(f"No output directories found for exp '{exp}' and seq '{seq}'")
    latest_dir = max(
        filtered_dirs,
        key=lambda d: pd.to_datetime(
            "_".join(d.split("_")[-2:]),
            format="%Y%m%d_%H%M%S"
        )
    )
    print("we found that the latest dir was ", latest_dir)
    return os.path.join(base_output_dir, latest_dir)


def encoder(npz_path = None, exp=None, seq=None):
    df = saved_npz_to_df(npz_path)
    num_gaussians = len(df)
    sidelen = int(np.sqrt(num_gaussians))

    #sorted_df = prune_gaussians(df, sidelen * sidelen)

    #### SORTING THE POINT CLOUD
    t1= time.time()   
    sorted_df = sort_dyn_gaussians(df, resume_from_last=False, exp=exp, seq=seq)
    t2 = time.time()
    print("sorting took ", t2 - t1, " seconds")

    ### COMPRESSING THE SORTED POINT CLOUD
    output_dir = './playground/compressed_outputs'+'/'+'compression_'+exp+"_"+seq+"_"+str(pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    # FFmpeg input: pipe frames as rawvideo (12-bit grayscale little-endian)

    params = df_to_params(sorted_df)
    print("params type ", type(params))
    # TODO: Convert to 12 bits little endian stored on 16 bits (gray12le) and save each attribute as a separate video. 
    # for each attribute of params
    # 1) Find the highest and lowest values of the attribute across all frames and all gaussians. This will be used to normalize the values to the 12 bit range (0-4095)
    os.makedirs(output_dir, exist_ok=False)
    data = {"key_list": []}
    Timesteps = params["means3D"].shape[0]
    for k, v in params.items():
        v = v.detach().cpu().numpy()
        print("processing : ", k)
        data["key_list"].append(k)
        max_ = v.max()
        min_ = v.min()
        data[f"{k}_max"] = float(max_)
        data[f"{k}_min"] = float(min_)

        if max_ != min_:
            frames = ((v - min_) / (max_ - min_)) * (2**12 - 1)
        else:
            print("min==max we have a problem")
            raise ValueError
        frames = np.clip(frames, 0, 2**12 - 1).astype(np.uint16)
        frames = frames.reshape((-1, sidelen, sidelen))
        D, H, W = frames.shape #Depth, Height, Width
        print("frames type", type(frames))
        print("frames shape", frames.shape)
        

        output_path = output_dir+f"/{k}_output_12bit.mp4"

        process = (
            ffmpeg
            .input(
                'pipe:',
                format='rawvideo',
                pix_fmt='gray12le',
                s=f'{sidelen}x{sidelen}',
                r=30
            )
            .output(
                output_path,
                pix_fmt='gray12le',
                vcodec='libx265',
                **{
                    'x265-params': 'preset=medium:lossless=1'
                }
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        # Write frames
        data[f"{k}_Depth"] = D
        for i, frame in enumerate(frames):
            process.stdin.write(frame.astype('<u2').tobytes())

        process.stdin.close()
        process.wait()


    print("key list ", data["key_list"])
    data["sidelen"] = sidelen
    data[f"timesteps"] = Timesteps
    with open(f"{output_dir}/compression_meta_data.json", "w") as f:
        json.dump(data, f)

    print("compression output folder :", output_dir)
    total_size = 0
    for file in os.listdir(output_dir):
        file_path= os.path.join(output_dir, file)
        file_size =os.path.getsize(file_path)
        total_size += file_size
    print("total compressed size (bytes) : ", total_size)
    return params

import subprocess

def decode_h265_to_frames(file_path, expected_shape):
    Depth, H, W = expected_shape
    cmd = [
        'ffmpeg',
        '-i', file_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'gray12le',
        '-'
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_frames, _ = process.communicate()
    frames = np.frombuffer(raw_frames, dtype='<u2').reshape((Depth, H, W))
    return frames


def decoder(exp=None, seq=None):
    compressed_output_dir = get_latest_output_dir(exp, seq)
    with open(os.path.join(compressed_output_dir, "compression_meta_data.json"), 'r') as f :
        data = json.load(f)
    sidelen = data["sidelen"]
    timesteps = data["timesteps"]
    params = {}
    for key in data["key_list"]:
        Depth = data[f"{key}_Depth"]
        max_ = np.float32(data[f"{key}_max"])
        min_ = np.float32(data[f"{key}_min"]) 
        file_path = os.path.join(compressed_output_dir, f"{key}_output_12bit.mp4")
        frames = decode_h265_to_frames(file_path, (Depth, sidelen,sidelen))
        try:
            frames = frames.reshape((timesteps, sidelen * sidelen, -1))
        except ValueError:
            frames = frames.reshape((sidelen * sidelen, -1))
        print("frame shape ", frames.shape)
        frames = np.float32(frames * (max_ - min_) / (2**12 - 1) + min_)
        params[key]=frames

    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(False).detach()) for k, v in
          params.items()}
    
    return params

if __name__ == "__main__":
    decompress_df()