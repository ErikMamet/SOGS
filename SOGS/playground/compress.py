import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json 
import time

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

def codec(npz_path = None, compress = False, use_sorted_indexes = False ,decompress = False, test_bypass= False, exp=None, seq=None):
    #date and time stamp for output folder naming
    print("Entering codec function ")
    if compress == True:
        output_dir = './playground/compressed_outputs'+'/'+'compression_'+exp+"_"+seq+"_"+str(pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)
        df = saved_npz_to_df(npz_path)
        if test_bypass == True :
            print("============================== NO COMPRESSION OR DECOMPRESSION =====================================")
            return df
        print("DONE CREATING DF")
        #write_path = os.path.join(output_dir, "concatenated_output.ply")
        #save_output_to_ply(concatenated_array, names, write_path, ascii=True)
        # `sort_dyn_gaussians` is a Click command when imported from
        # `sort_dyn_3d_gaussians.py` (it is decorated with @click.command()).
        # Calling it directly will make Click try to parse sys.argv which causes
        # the "unexpected extra arguments" error. Call the underlying callback
        # (the original Python function) when available.
        print("ENTERING SORTING")     
        t1= time.time()   
        sorted_df = sort_dyn_gaussians(df, resume_from_last=use_sorted_indexes)
        t2 = time.time()
        print("sorting took ", t2 - t1, " seconds")
        print("ENTERING COMPRESSION")
        compress_df(sorted_df, out_folder_path=output_dir)
        t3 = time.time()
        print("compression took ", t3 - t2, " seconds")
        print("compression output folder :", output_dir)
        total_size = 0
        for file in os.listdir(output_dir):
            file_path= os.path.join(output_dir, file)
            file_size =os.path.getsize(file_path)
            total_size += file_size
        print("total compressed size (bytes) : ", total_size)
    if decompress == True:
        #df = saved_npz_to_df(npz_path)
        #return prune_gaussians(df, int(np.sqrt(len(df)))**2)
        output_dir = get_latest_output_dir(exp, seq)
        print("++++++++ decompressing ", output_dir)
        total_size = 0
        for file in os.listdir(output_dir):
            file_path= os.path.join(output_dir, file)
            file_size =os.path.getsize(file_path)
            total_size += file_size
        print("total compressed size (bytes) : ", total_size)
        decompressed_df = decompress_df(output_dir)
        #print decompressed_df keys
        # compare decompressed df to original df
        # make sure they have the same shape and columns
        # make sure that column values are close enough (since compression could be lossy)
        if npz_path is not None:
            df = saved_npz_to_df(npz_path)
            sorted_df = sort_dyn_gaussians(df, resume_from_last=use_sorted_indexes)
            report = compare_dataframes(sorted_df, decompressed_df, json_path=get_latest_output_dir(exp, seq) + "/decompression_report.json")
        print("WE ARE RETURNING THE SORTED_DF DF AS A TEST")
        return decompressed_df
    #TODO, understand (no chat gpt) and modify core.py so that is takes in vectors of any size (since ours are bound to vary)
    
if __name__ == "__main__":
    decompress_df()