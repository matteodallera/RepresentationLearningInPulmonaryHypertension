import numpy as np
import SimpleITK as sitk
import torch
from transformers import AutoModel, AutoImageProcessor
import SimpleITK as sitk
import numpy as np
import os
from monai.inferers import SlidingWindowInferer
from monai.transforms import Resize
import torch.nn as nn
import pandas as pd
import re
import time

def label_fn(fname):
    return 1 if "pah" in fname.lower() else 0

def predictor_fn(x):
    # Reshape the patch tokens to resemble a 3D feature map
    out = model(x, reshape=True)
    return out.last_hidden_state


def resizing(volume, spatial_size):
    resize = Resize(spatial_size, mode="trilinear", align_corners=False)
    volume = volume[None, ...]
    x = torch.tensor(volume)
    x = resize(x)
    x = x.squeeze(0).numpy()
    return x


def sort_key(nome):
    match = re.match(r"(CTEPH|PAH)(\d+)", nome)
    if match:
        tipo = match.group(1)
        numero = int(match.group(2))
        return (0 if tipo == "CTEPH" else 1, numero)
    return (2, nome)  # fallback


inferer = SlidingWindowInferer(
    roi_size=[12, 224, 224],
    sw_batch_size=5,
    overlap=0.75,
    mode='gaussian'
)


device = 'cuda'

# Load the model
model = AutoModel.from_pretrained('fomofo/tap-ct-b-3d', trust_remote_code=True)
preprocessor = AutoImageProcessor.from_pretrained('fomofo/tap-ct-b-3d', trust_remote_code=True)
model.to(device)
model.eval()



reader = sitk.ImageSeriesReader()
FeatureMatrix = [ ]
label = [ ]
forward_times = [ ]
peaks_memory = [ ]
num_params = sum(p.numel() for p in model.parameters())

Dataset = r"/home/matteo_dallera/Desktop/MatteoDATA/PaperRapLearning/Dati_allenamento_modelli/Dati_San_Matteo_original"
pazienti = sorted(os.listdir(Dataset), key=sort_key)
num = 0
start_time = time.perf_counter()
for paz in pazienti:
    print(num)
    CurrentPaz = os.path.join(Dataset, paz)
    dicom_names = reader.GetGDCMSeriesFileNames(CurrentPaz)

    reader.SetFileNames(dicom_names)
    Vol = reader.Execute()
    #Vol = sitk.ReadImage(CurrentPaz)
    Vol = sitk.DICOMOrient(Vol, 'LPS')
    array = sitk.GetArrayFromImage(Vol)
    array = resizing(array,[array.shape[0],array.shape[1],array.shape[2]])
    array = np.expand_dims(array, axis=(0, 1))

    x = preprocessor(array)['pixel_values']
    x=x.to('cuda')
    y= label_fn(paz)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output = inferer(x, predictor_fn)


    adaptive_gap = nn.AdaptiveAvgPool3d((1, 1, 1))
    output = adaptive_gap(output)
    output = output.flatten(1)

    torch.cuda.synchronize()
    forward_times.append(time.perf_counter() - t0)
    peaks_memory.append(torch.cuda.max_memory_allocated(device) / (1024**2))
    torch.cuda.empty_cache()
    FeatureMatrix.append(output)
    label.append(y)

    num+=1






FeatureMatrix = torch.cat(FeatureMatrix, dim=0).cpu().numpy()
num_features = FeatureMatrix.shape[1]
columns = [f"feature_{i+1}" for i in range(num_features)]

df = pd.DataFrame(FeatureMatrix, columns=columns)
df["label"] = label

# Saving Embeddings to perform the downstream analysis
output_path = "/home/matteo_dallera/Desktop/alcune_analisi/tap.xlsx"
df.to_excel(output_path, index=False)

