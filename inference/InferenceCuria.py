import numpy as np
import SimpleITK as sitk
import torch
from transformers import AutoModel, AutoImageProcessor
import SimpleITK as sitk
import numpy as np
import os
# Forward pass with sliding window
from monai.inferers import SlidingWindowInferer
from monai.transforms import Resize
import torch.nn as nn
import pandas as pd
from utilsCuria import extract_3D_features_avg_per_volume
import time
import pandas as pd



def label_fn(fname):
    return 1 if "pah" in fname.lower() else 0


def curia(x, model):
    with torch.no_grad():
        output = model(x,output_hidden_states=True)
        feature = extract_3D_features_avg_per_volume(output)

    return feature


def resizing(volume, spatial_size):
    resize = Resize(spatial_size, mode="trilinear", align_corners=False)
    volume = volume[None, ...]
    x = torch.tensor(volume)
    x = resize(x)
    x = x.squeeze(0).numpy()
    return x

import re

def sort_key(nome):
    match = re.match(r"(CTEPH|PAH)(\d+)", nome)
    if match:
        tipo = match.group(1)
        numero = int(match.group(2))
        return (0 if tipo == "CTEPH" else 1, numero)
    return (2, nome)  # fallback



from transformers import AutoModel
model = AutoModel.from_pretrained("raidium/curia")
from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained("raidium/curia", trust_remote_code=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to('cuda')
model.eval()

# Setting Device to compute the Peak Memory Usage and printing the number of parameters

forward_times = []
peaks_memmory = []
num_params = sum(p.numel() for p in model.parameters())
print(num_params)


reader = sitk.ImageSeriesReader()


# Defining the Feature Matrix and label list
FeatureMatrix = [ ]
label = [ ]

# loading Data to make inference on
Dataset = r"/home/matteo_dallera/Desktop/MatteoDATA/PaperRapLearning/Dati_allenamento_modelli/Dati_San_Matteo_original"

pazienti = sorted(os.listdir(Dataset), key=sort_key)




num = 0
for paz in pazienti:
# setting a counter
    print(num)

    CurrentPaz = os.path.join(Dataset, paz)
    dicom_names = reader.GetGDCMSeriesFileNames(CurrentPaz)
    reader.SetFileNames(dicom_names)
    Vol = reader.Execute()


# preparing the volume to make inference on

    Vol = sitk.DICOMOrient(Vol, 'LPS')
    array = sitk.GetArrayFromImage(Vol)
    array = np.transpose(array, (1, 2, 0))
    array = resizing(array,[array.shape[0],array.shape[1],array.shape[2]])


    x = processor(array)
    x=x.to('cuda')
    y= label_fn(paz)


    torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    feature = curia(x.pixel_values[0], model)

    torch.cuda.synchronize()
    forward_times.append(time.perf_counter() - t0)
    peaks_memmory.append(torch.cuda.max_memory_allocated(device) / (1024**2))

    FeatureMatrix.append(feature.cpu())
    label.append(y)

# GPU memory need to be cleaned
    del x
    torch.cuda.empty_cache()
    num+=1




FeatureMatrix = torch.stack(FeatureMatrix, dim=0).numpy()
num_features = FeatureMatrix.shape[1]
columns = [f"feature_{i+1}" for i in range(num_features)]

df = pd.DataFrame(FeatureMatrix, columns=columns)
df["label"] = label

# Saving Excel file in the following format : [embeddings as rows, last column is the label]
output_path = "/home/matteo_dallera/Desktop/PossibiliDomande_reviewer/CuriaMIP.xlsx"
df.to_excel(output_path, index=False)


