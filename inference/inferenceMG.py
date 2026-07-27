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

weight_dir = ""

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


path_vol = r""
cartelle = sorted(os.listdir(path_vol), key=sort_key)




resize = Resize(spatial_size=(24,256,256), mode="trilinear", align_corners=False)
#resize = Resize(spatial_size=(128,256,256), mode="trilinear", align_corners=False)

Label = [ ]
FeatureMatrix = [ ]
target_model.eval()
forward_times = [ ]
peaks_memory = [ ]
classe = [ ]

for j in range(len(cartelle)):
    print(j)
    CurrentVolPath = os.path.join(path_vol, cartelle[j])
    CurrentVol = sitk.ReadImage(CurrentVolPath)
    arr = sitk.GetArrayFromImage(CurrentVol).astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr[None, ...]
    x = torch.tensor(arr)
    x = resize(x).to(device)
    y1 = label_fn(cartelle[j])

    classe.append(y1)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    pred,detail = target_model(x.unsqueeze(0))

    torch.cuda.synchronize()
    forward_times.append(time.perf_counter() - t0)
    peaks_memory.append(torch.cuda.max_memory_allocated(device) / (1024**2))


    pred = pred.detach().cpu().numpy()
    detail = detail.detach().cpu().numpy()
    final = np.concat((pred,detail), axis=1)
    FeatureMatrix.append(final)





FeatureMatrix = np.vstack(FeatureMatrix)
num_features = FeatureMatrix.shape[1]
columns = [f"feature_{i+1}" for i in range(num_features)]
df = pd.DataFrame(FeatureMatrix, columns=columns)
df["classe"] = classe

output_path = ""




