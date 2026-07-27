import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unetmod
from torchinfo import summary
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torchinfo import summary
import os
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # opzionale, per leggere immagini
import gc
from monai.transforms import Resize
import matplotlib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import umap
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import f1_score
import sys
import time
import pandas as pd

# Carica il file Excel
df = pd.read_excel("/home/matteo_dallera/PycharmProjects/ModelG2/StatisticheDataset/patient_kvp.xlsx")


def label_fn(fname):
    return 1 if "pah" in fname.lower() else 0



class TargetNet(nn.Module):
    def __init__(self, base_model, n_class=1):
        super(TargetNet, self).__init__()

        self.base_model = base_model
        #self.dense_1 = nn.Linear(512, 1024, bias=True)
        #self.dense_2 = nn.Linear(1024, n_class, bias=True)


    def forward(self, x):
        self.base_model(x,1)
        self.base_out = self.base_model.out512
        self.base_detail = self.base_model.skip_out128
        # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
        # where N = batch_size, C = channels, H = height, and W = Width
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
        self.out_glb_avg_pool2 = F.avg_pool3d(self.base_detail, kernel_size=self.base_detail.size()[2:]).view(self.base_detail.size()[0], -1)
        final_out = self.out_glb_avg_pool
        final_detail = self.out_glb_avg_pool2
        #self.linear_out = self.dense_1(self.out_glb_avg_pool)
        #final_out = self.dense_2( F.relu(self.linear_out))
        return final_out, final_detail

import re

def sort_key(nome):
    match = re.match(r"(CTEPH|PAH)(\d+)", nome)
    if match:
        tipo = match.group(1)
        numero = int(match.group(2))
        return (0 if tipo == "CTEPH" else 1, numero)
    return (2, nome)  # fallback


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = unetmod.UNet3D()


# Load pre-trained weights
#weight_dir = './pesi/Genesis_Chest_CT_nuovi4.pt'
#weight_dir = './pesi_MICCAI/Genesis_Chest_CT_veri.pt'
#weight_dir = './pesi_MICCAI/Genesis_Chest_final_PREP.pt'
#weight_dir = './pesi_MICCAI/Genesis_Chest_final_NO_PREP.pt'
#weight_dir = "./pesi_paper/Nova_prep_final.pt"
#weight_dir = "./pesi_paper/Nova_no_prep_final.pt"
#weight_dir = "./pesi_paper/Nova_no_prep_final_0bias_prep.pt"
weight_dir = "./pesi_paper/Nova_no_prep_final_0bias.pt"

checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
base_model.load_state_dict(unParalled_state_dict)

# definition of the Model
target_model = TargetNet(base_model)
target_model.to(device)

num_params = sum(p.numel() for p in target_model.parameters())
print(num_params)


path_vol = r"/home/matteo_dallera/Desktop/MatteoDATA/PaperRapLearning/Dati_allenamento_modelli/Dati_Processed/Intermedio"
#cartelle = os.listdir(path_vol)
cartelle = sorted(os.listdir(path_vol), key=sort_key)

with open("nomi.txt", "w") as f:
    for nome in cartelle:
        f.write(nome + "\n")


# it's useful to interpolate the volume to fixed shape [64 x 64 x 64 ] (I feel like it's way too much information loss, better spatial resolutions lead to better performance (?))
#resize = Resize(spatial_size=(24,256,256), mode="trilinear", align_corners=False)
resize = Resize(spatial_size=(128,256,256), mode="trilinear", align_corners=False)

Label = [ ]
FeatureMatrix = [ ]
target_model.eval()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)


classe = [ ]
start_time = time.perf_counter()
for j in range(len(cartelle)):
    print(j)
    CurrentVolPath = os.path.join(path_vol, cartelle[j])
    match = re.match(r"^([A-Za-z]+_?\d+)", cartelle[j])

    if match:
        patient_id = match.group(1)

    print(patient_id)
    row = df[df["id"] == patient_id]
    kvp = row["kvp"].iloc[0]

    CurrentVol = sitk.ReadImage(CurrentVolPath)
    arr = sitk.GetArrayFromImage(CurrentVol).astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr[None, ...]
    x = torch.tensor(arr)
    x = resize(x).to(device)
    y1 = label_fn(cartelle[j])
    y = kvp
    classe.append(y1)

    pred,detail = target_model(x.unsqueeze(0))
    pred = pred.detach().cpu().numpy()
    detail = detail.detach().cpu().numpy()
    final = np.concat((pred,detail), axis=1)
    Label.append(y)
    FeatureMatrix.append(final)


end_time = time.perf_counter()

inference_time = end_time - start_time
print(f"Tempo di inferenza: {inference_time:.6f} secondi")


# Converti a matrice NumPy
FeatureMatrix = np.vstack(FeatureMatrix)

# Crea DataFrame: ogni feature diventa una colonna
num_features = FeatureMatrix.shape[1]
columns = [f"feature_{i+1}" for i in range(num_features)]

df = pd.DataFrame(FeatureMatrix, columns=columns)
#df["label"] = Label
df["classe"] = classe

# Salva in Excel
output_path = "/home/matteo_dallera/Desktop/CV_RAE_buttare.xlsx"
#df.to_excel(output_path, index=False)

peak_memory_bytes = torch.cuda.max_memory_allocated(device)
peak_memory_mb = peak_memory_bytes / (1024 ** 2)  # Converte in MB

print(f"Peak GPU Memory Usage: {peak_memory_mb:.2f} MB")




