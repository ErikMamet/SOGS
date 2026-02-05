import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

def saved_npz_to_df(npz_path):
    """
    Convert params.npz into a flat DataFrame.
    Rows = points (N)
    Columns = time-varying params flattened by timestep
    """
    data = np.load(npz_path)

    blocks = []  # collect DataFrame blocks here

    # ------------------------
    # means3D: (T, N, 3)
    # ------------------------
    if "means3D" in data:
        means = data["means3D"]
        T, N, C = means.shape
        assert C == 3

        arr = means.transpose(1, 0, 2).reshape(N, 3 * T)
        cols = [f"{axis}_{t}" for t in range(T) for axis in ("x", "y", "z")]

        blocks.append(pd.DataFrame(arr, columns=cols))

    else:
        raise KeyError("means3D not found in npz")

    # ------------------------
    # rgb_colors: (T, N, 3)
    # ------------------------
    if "rgb_colors" in data:
        rgbs = data["rgb_colors"]
        T2, N2, C = rgbs.shape
        assert (T2, N2, C) == (T, N, 3)

        arr = rgbs.transpose(1, 0, 2).reshape(N, 3 * T)
        cols = [f"{axis}_{t}" for t in range(T) for axis in ("r", "g", "b")]

        blocks.append(pd.DataFrame(arr, columns=cols))

    # ------------------------
    # unnorm_rotations: (T, N, 4)
    # ------------------------
    if "unnorm_rotations" in data:
        rotations = data["unnorm_rotations"]
        T2, N2, C = rotations.shape
        assert (T2, N2, C) == (T, N, 4)

        arr = rotations.transpose(1, 0, 2).reshape(N, 4 * T)
        cols = [f"q{q}_{t}" for t in range(T) for q in range(4)]

        blocks.append(pd.DataFrame(arr, columns=cols))

    # ------------------------
    # logit_opacities: (N,)
    # ------------------------
    if "logit_opacities" in data:
        blocks.append(pd.DataFrame({
        "alpha": data["logit_opacities"].squeeze()
        }))


    # ------------------------
    # log_scales: (N, 3)
    # ------------------------
    if "log_scales" in data:
        scales = data["log_scales"]
        assert scales.shape == (N, 3)

        blocks.append(pd.DataFrame({
            "s0": scales[:, 0],
            "s1": scales[:, 1],
            "s2": scales[:, 2],
        }))

    # ------------------------
    # Final concat
    # ------------------------
    df = pd.concat(blocks, axis=1)
    return df


import os
import numpy as np
import pandas as pd


def df_to_params(df):
    """
    Reconstruct numpy arrays from DataFrame.
    Returns a pure NumPy dict.
    """

    params = {}

    # ------------------------
    # means3D
    # ------------------------
    x_cols = sorted([c for c in df.columns if c.startswith("x_")],
                    key=lambda c: int(c.split("_")[1]))
    y_cols = sorted([c for c in df.columns if c.startswith("y_")],
                    key=lambda c: int(c.split("_")[1]))
    z_cols = sorted([c for c in df.columns if c.startswith("z_")],
                    key=lambda c: int(c.split("_")[1]))

    if x_cols:
        x = np.stack([df[c].values for c in x_cols], axis=0)
        y = np.stack([df[c].values for c in y_cols], axis=0)
        z = np.stack([df[c].values for c in z_cols], axis=0)
        params["means3D"] = np.stack([x, y, z], axis=2)  # (T, N, 3)

    # ------------------------
    # rgb_colors
    # ------------------------
    r_cols = sorted([c for c in df.columns if c.startswith("r_")],
                    key=lambda c: int(c.split("_")[1]))
    g_cols = sorted([c for c in df.columns if c.startswith("g_")],
                    key=lambda c: int(c.split("_")[1]))
    b_cols = sorted([c for c in df.columns if c.startswith("b_")],
                    key=lambda c: int(c.split("_")[1]))

    if r_cols:
        r = np.stack([df[c].values for c in r_cols], axis=0)
        g = np.stack([df[c].values for c in g_cols], axis=0)
        b = np.stack([df[c].values for c in b_cols], axis=0)
        params["rgb_colors"] = np.stack([r, g, b], axis=2)  # (T, N, 3)

    # ------------------------
    # unnorm_rotations
    # ------------------------
    q_cols = [
        sorted([c for c in df.columns if c.startswith(f"q{i}_")],
               key=lambda c: int(c.split("_")[1]))
        for i in range(4)
    ]

    qs = [np.stack([df[c].values for c in qc], axis=0) for qc in q_cols]
    params["unnorm_rotations"] = np.stack(qs, axis=2)  # (T, N, 4)

    # ------------------------
    # log_opacities
    # ------------------------
    if "alpha" in df.columns:
        params["logit_opacities"] = df["alpha"].values

    # ------------------------
    # log_scales
    # ------------------------
    if {"s0", "s1", "s2"}.issubset(df.columns):
        params["log_scales"] = np.stack(
            [df["s0"].values, df["s1"].values, df["s2"].values], axis=1
        )

    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in params.items() if v is not None}

    return params


# Example usage:folder_path = 'path/to/your/folder'
if __name__ == "__main__":
    folder_path = '/home/erikmamet/Current_proj/Dynamic3DGaussians/output/PanopticSports_baseline/boxes/params.npz'
    df = saved_npz_to_df(folder_path)
    #output_dir = '/home/erik/Documents/Self-Organizing-Gaussians/playground/outputs'
    #write_path = os.path.join(output_dir, "concatenated_output.ply")
    ##save_output_to_ply(concatenated_array, names, write_path, ascii=True)
