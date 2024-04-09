import numpy as np
import pandas as pd
import torch
import math
from utils.vae_utils import initialize_vae, vae_seqs_to_zs


def load_molecule_train_data(
    task_id,
    gen_z_bsz=128,
): 
    df = pd.read_csv("init_data/guacamol_train_data_first_20k.txt") # (20000, 14)
    x_selfies = df['selfie'].values.tolist() 
    all_y = torch.from_numpy(df[task_id].values).float()  # torch.Size([20000])
    all_z = load_train_z(
        all_x=x_selfies,
        task_id=task_id,
        gen_z_bsz=gen_z_bsz,
    ) 
    return all_z, all_y

def load_train_z(
    all_x,
    task_id,
    gen_z_bsz=64,
):
    path_to_init_train_zs = f"init_data/all_zs.pt"
    # if we have pre-computed train zs for vae, load them
    try:
        zs = torch.load(path_to_init_train_zs)
    # otherwise, compute zs with vae 
    except: 
        zs = compute_train_zs(
            all_x=all_x,
            bsz=gen_z_bsz,
            path_save_zs=path_to_init_train_zs,
        )

    return zs

def compute_train_zs(
    all_x,
    path_save_zs,
    bsz=64
):  
    vae, dataobj = initialize_vae()
    init_zs = []
    n_batches = math.ceil(len(all_x)/bsz)
    for i in range(n_batches):
        xs_batch = all_x[i*bsz:(i+1)*bsz] 
        zs = vae_seqs_to_zs(vae=vae, dataobj=dataobj, selfies_list=xs_batch)
        init_zs.append(zs.detach().cpu())
    init_zs = torch.cat(init_zs, dim=0)
    # now save the zs so we don't have to recompute them in the future:
    torch.save(init_zs, path_save_zs)

    return init_zs