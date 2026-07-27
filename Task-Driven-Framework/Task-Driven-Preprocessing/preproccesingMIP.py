from fileinput import filename
import pydicom
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from monai.transforms import SpatialPad, Resize


import torch

def block_max(vol: torch.Tensor, S: int) -> torch.Tensor:
    """
    vol: tensor di shape (N, H, W)
    S: dimensione dei blocchi lungo N
    ritorna: tensor di shape (num_blocks, H, W)
    """
    N, H, W = vol.shape
    num_blocks = (N + S - 1) // S

    result = []
    for i in range(0, N, S):
        block = vol[i:i+S]
        result.append(block.max(dim=0).values)

    return torch.stack(result)



path_vol = r""
path_out = r""
cartelle = [nome for nome in os.listdir(path_vol) ]
numero_cartelle = len(cartelle)
sh =[ ]
resize = Resize(spatial_size=(240, 256, 256), mode='trilinear', align_corners=False)

for j in range(numero_cartelle):
    CurrentVold_ind = os.path.join(path_vol, cartelle[j])
    ref = sitk.ReadImage(CurrentVold_ind)
    CurrentVol = sitk.GetArrayFromImage(ref)
    CurrentVol = torch.from_numpy(CurrentVol)
    CurrentVol = CurrentVol.unsqueeze(0)

    sh.append(CurrentVol.shape[1])

    CurrentVol = resize(CurrentVol)
    vol = CurrentVol[0, :, :, :]

    vol_finale = block_max(vol, 10)


    image_finale = sitk.GetImageFromArray(vol_finale)
    image_finale.SetSpacing(ref.GetSpacing())
    image_finale.SetOrigin(ref.GetOrigin())
    image_finale.SetDirection(ref.GetDirection())

    output_path1 = path_out + "/"+ cartelle[j]

    #sitk.WriteImage(image_finale, output_path1, useCompression=False)
    print(f"percentuale di completamento: {round(j/numero_cartelle*100,2)}%")