import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


def saved_npz_to_df(npz_path):
    """
    Concatenates all .npy files in a folder that match the shape of means3D.npy.

    Args:
        folder_path (str): Path to the folder containing .npy files.
        means3D_path (str): Path to the means3D.npy file.

    Returns:
        np.ndarray: Concatenated array including all matching arrays.
    """
    # Starting coordinates
    npz_dict = np.load(npz_path)
    names = []

    # Load the reference means3D array
    #means3D_path = os.path.join(folder_path, "means3D.npy")
    means3D = npz_dict['means3D']
    print("1) means3D-shape : ", means3D.shape)
    means3D = np.transpose(means3D, (0, 2, 1))
    print("2) means3D-shape after column swap : ", means3D.shape)
    means3D = means3D.transpose() # so that all x (resp y, resp z) coordinates are grouped together when flattened 
    print("3) means3D-shape after column swap + transpose: ", means3D.shape)
    print(means3D)
    num_points = means3D.shape[0] # number of points in the PC
    num_timesteps = means3D.shape[2] # number of points in the PC
    output = means3D.reshape((num_points,3*num_timesteps)) # the x coordinates are grounpes 
    # print(means3D[0,:156])

    # name the position at additional time steps 
    for i in range(means3D.shape[2]):
        names.append("x_"+str(i))
    for i in range(means3D.shape[2]):
        names.append("y_"+str(i))
    for i in range(means3D.shape[2]):
        names.append("z_"+str(i))

    print("Number of points in means3D:", num_points)

    # List all .npy files in the folder
    #npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    # Iterate through the files and concatenate any array that can be reshaped
    # to (num_points, -1). Skip the reference file itself.
    #for file in npy_files:
    
    print("these are the npz dict keys", npz_dict.keys())
    
    for k in npz_dict.keys():
        #ignore the segmentation of the file 
        if k == 'seg_colors':
            #skip seg_colors.npy files
            continue

        # skip the reference file
        if k == 'means3D':
            continue
        
        #for now print the shape of the files 
        try:
            data = npz_dict[k]
            print(type(data))
            if data.ndim == 3:
                data = np.transpose(data, (0, 2, 1))
            data = data.transpose() # so that all x (resp y, resp z) coordinates are grouped together when flattened 

            print(f"Checking field: {k} with shape {data.shape}")
            # Try to reshape `data` to (num_points, -1)
            try:
                reshaped = data.reshape((num_points, -1))
            except Exception:
                print(f"Cannot reshape {k} to ({num_points}, -1); skipping")
                continue

            # If reshape succeeded, concatenate along axis=1 (columns are features)
            if reshaped.shape[0] == num_points:
                output = np.concatenate((output, reshaped), axis=1)
                    #name variables 
                
                if k == "logit_opacities":
                    for i in range(reshaped.shape[1]):
                        names.append(f"alpha")
                if k == "log_scales":
                    print('for scale reshaped.shape[1]', reshaped.shape[1])
                    names+=["s0","s1","s2"]
                if k == "unnorm_rotations":
                    names+=[f"q0_{i}" for i in range(reshaped.shape[1]//4)]
                    names+=[f"q1_{i}" for i in range(reshaped.shape[1]//4)]
                    names+=[f"q2_{i}" for i in range(reshaped.shape[1]//4)]
                    names+=[f"q3_{i}" for i in range(reshaped.shape[1]//4)]
                if k == "rgb_colors":
                    names+=[f"r_{i}" for i in range(reshaped.shape[1]//3)]
                    names+=[f"g_{i}" for i in range(reshaped.shape[1]//3)]
                    names+=[f"b_{i}" for i in range(reshaped.shape[1]//3)]

                print(f"Reshaped {k} to {reshaped.shape}, concatenating. ", len(names))

                print(f"Appended {k} as shape {reshaped.shape}; output now {output.shape}")
            else:
                print(f"Reshaped {k} has unexpected rows {reshaped.shape[0]}; skipping")
        except Exception as e:
            print(f"Skipping {k}: {e}")
    #print("Final concatenated shape:", output.shape, names)
    # Convert concatenated array and names to a pandas DataFrame
    df = pd.DataFrame(output, columns=names)
    # Print the names of the DataFrame columns
    #print("DataFrame columns:")
    #print(df.columns.tolist())
    #print(df.loc[:, df.columns.str.startswith("x")].values[0,:].shape)

    return df

import os
import numpy as np
import pandas as pd


def df_to_params(df, output_folder):
    """
    Reconstructs the original numpy arrays saved in saved_npy_to_df and writes
    them back into .npy files inside output_folder.

    Args:
        df (pd.DataFrame): DataFrame created by saved_npy_to_df
        output_folder (str): Where to save reconstructed npz files
    """


    # ---------------------------------------------------
    # 1. Reconstruct means3D.npy
    # ---------------------------------------------------
    # Columns: x_0 ... x_T, y_0 ... y_T, z_0 ... z_T
    x_cols = [c for c in df.columns if c.startswith("x_")]
    y_cols = [c for c in df.columns if c.startswith("y_")]
    z_cols = [c for c in df.columns if c.startswith("z_")]

    x_vals = df[x_cols].values    # (N, T)
    y_vals = df[y_cols].values
    z_vals = df[z_cols].values

    # Shape back to (N, 3, T) then transpose to original orientation if needed
    means3D = np.stack([x_vals, y_vals, z_vals], axis=1)  # (N, 3, T)
    means3D = np.transpose(means3D, (0, 2, 1))            # (N, T, 3)

    # ---------------------------------------------------
    # 2. logit_opacities.npy (alphas)
    # ---------------------------------------------------
    alpha_cols = [c for c in df.columns if c == "alpha"]
    if len(alpha_cols) > 0:
        alphas = df[alpha_cols].values

    # ---------------------------------------------------
    # 3. log_scales.npy (s0, s1, s2)
    # ---------------------------------------------------
    scale_cols = [c for c in df.columns if c in ["s0", "s1", "s2"]]
    if len(scale_cols) == 3:
        scales = df[scale_cols].values  # (N, 3)
        # reshape to original (N, 1, 3)
        scales = np.transpose(scales.reshape((-1, 3, 1)), (0, 2, 1))

    # ---------------------------------------------------
    # 4. unnorm_rotations.npy  (quaternions)
    # q0_i, q1_i, q2_i, q3_i
    # ---------------------------------------------------
    q0_cols = sorted([c for c in df.columns if c.startswith("q0_")],
                     key=lambda x: int(x.split("_")[1]))
    q1_cols = sorted([c for c in df.columns if c.startswith("q1_")],
                     key=lambda x: int(x.split("_")[1]))
    q2_cols = sorted([c for c in df.columns if c.startswith("q2_")],
                     key=lambda x: int(x.split("_")[1]))
    q3_cols = sorted([c for c in df.columns if c.startswith("q3_")],
                     key=lambda x: int(x.split("_")[1]))

    if len(q0_cols) > 0:
        q0 = df[q0_cols].values
        q1 = df[q1_cols].values
        q2 = df[q2_cols].values
        q3 = df[q3_cols].values

        R = np.stack([q0, q1, q2, q3], axis=2)  # (N, T, 4)
        R = np.transpose(R, (0, 2, 1))         # (N, 4, T)

    # ---------------------------------------------------
    # 5. rgb_colors.npy
    # r_i, g_i, b_i
    # ---------------------------------------------------
    r_cols = sorted([c for c in df.columns if c.startswith("r_")],
                    key=lambda x: int(x.split("_")[1]))
    g_cols = sorted([c for c in df.columns if c.startswith("g_")],
                    key=lambda x: int(x.split("_")[1]))
    b_cols = sorted([c for c in df.columns if c.startswith("b_")],
                    key=lambda x: int(x.split("_")[1]))

    if len(r_cols) > 0:
        r = df[r_cols].values
        g = df[g_cols].values
        b = df[b_cols].values

        colors = np.stack([r, g, b], axis=2)  # (N, T, 3)
        colors = np.transpose(colors, (0, 2, 1))  # (N, 3, T)

    params = {
        'means3D': means3D,
        'rgb_colors': colors if len(r_cols) > 0 else None,
        'unnorm_rotations': R if len(q0_cols) > 0 else None,
        'logit_opacities': alphas if len(alpha_cols) > 0 else None,
        'log_scales': scales if len(scale_cols) == 3 else None,
        'cam_m' : np.load(os.path.join(output_folder, "cam_m.npy")), #camera exposure --> these need to be managed
        'cam_c': np.load(os.path.join(output_folder, "cam_c.npy")) #lighting offset --> these need to be managed
        #'seg_colors': None
    }
    
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(False).detach()) for k, v in
              params.items()}
    
    
    
    return params


# Example usage:folder_path = 'path/to/your/folder'
if __name__ == "__main__":
    folder_path = '/home/erikmamet/Current_proj/Dynamic3D-GS-Compression/output/PanopticSports_baseline/boxes/params.npz'
    df = saved_npz_to_df(folder_path)
    #output_dir = '/home/erik/Documents/Self-Organizing-Gaussians/playground/outputs'
    #write_path = os.path.join(output_dir, "concatenated_output.ply")
    ##save_output_to_ply(concatenated_array, names, write_path, ascii=True)
