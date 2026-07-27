import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
import matplotlib.image as mpimg
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
from torch import nn
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unetmod
import random


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed_all(seed)


best_loss = 40

from monai.transforms import Compose, RandAffine, Resize


class VolumeDataset(Dataset):
    def __init__(self, root, resize=None, augmentation=False):
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.endswith(".nii")])
        self.resize = resize
        self.augmentation = augmentation

        # Definisci i transform di augmentation
        if self.augmentation:
            self.augment = Compose([
                RandAffine(
                    prob=0.3,
                    rotate_range=(0, 0, np.pi / 36),  # Solo asse Z (assiale), ±5°
                    mode="trilinear",
                    padding_mode="zeros"
                )
            ])
        else:
            self.augment = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.root, fname)
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = arr[None, ...]
        x = torch.tensor(arr)
        x = self.resize(x)

        # Applica augmentation solo al training set
        if self.augment is not None:
            x = self.augment(x)

        y = x
        return x, y
#Declare the Dice Loss
def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_vol = r""
path_val = r""
resize = Resize(spatial_size=(24,256,256), mode="trilinear", align_corners=False)
data_ds = VolumeDataset(path_vol,resize, augmentation=True)
train_loader = DataLoader(data_ds, batch_size=8, shuffle=True)

validation_ds = VolumeDataset(path_val,resize, augmentation = False)  # Using a different validation set to monitor training's
val_loader = DataLoader(validation_ds, batch_size=8, shuffle=False)

model = unetmod.UNet3D()
#weight_dir = ''
#checkpoint = torch.load(weight_dir)
#print(checkpoint['epoch'])
#state_dict = checkpoint['state_dict']
#unParalled_state_dict = {}
#for key in state_dict.keys():
#    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
#model.load_state_dict(unParalled_state_dict)

model.to(device)
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), 1, momentum=0.9, weight_decay=0.0, nesterov=False)

# train the model
#training parameters : initial_epoch
TrainLOSS = []
ValLOSS = []
intial_epoch = 0
nb_epoch = 450
train_loss = 0

for epoch in range(intial_epoch, nb_epoch):

    model.train()
    for batch_ndx, (x,y) in enumerate(train_loader):
        print(f"\r{batch_ndx}/{len(train_loader)}", end='')
        x, y = x.float().to(device), y.float().to(device)
        optimizer.zero_grad()
        pred = model(x,1)


        loss = criterion(pred, y)
        train_loss += loss
        loss.backward()
        optimizer.step()


    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.float().to(device), y.float().to(device)
            pred = model(x,1)
            val_loss += criterion(pred, y).item()
            pred1 = model(x,0)
        # saving an image of a reconstructed slice in order to visually understand model training
        plt.imshow(abs(pred[0, 0, 12, :, :] - x[0, 0, 12, :, :]).detach().to('cpu'), cmap='gray')
        plt.savefig('diff')
        plt.imshow(pred[0, 0, 12, :, :].detach().to('cpu'), cmap='gray')
        plt.savefig('pred_skip')
        plt.imshow(pred1[0, 0, 12, :, :].detach().to('cpu'), cmap='gray')
        plt.savefig('pred_skip_senza')

    val_loss /= len(val_loader)
    train_loss /= len(train_loader)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')
    TrainLOSS.append(train_loss.detach().cpu().numpy())
    ValLOSS.append(val_loss)
    np.save('TrainLOSS2.npy', np.array(TrainLOSS))
    np.save('ValLOSS2.npy', np.array(ValLOSS))

    plt.figure(figsize=(8, 5))

    plt.plot(TrainLOSS, label='Train Loss')
    plt.plot(ValLOSS, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.legend()

    plt.grid(True)

    plt.savefig('loss_curve2.png')
    plt.close()


    if val_loss < best_loss:
        print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, val_loss))

        best_loss = val_loss
        num_epoch_no_improvement = 0
        # save model
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join("/MISTERG/pesi_paper", "Nova_no_prep_final_0bias_prep.pt"))
        print("Saving model ", os.path.join("/MISTERG/pesi_paper", "Nova_no_prep_final_0bias_prep.pt"))
        #inferencRUN.inference()
    else:
        print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,
                                                                                                  num_epoch_no_improvement))
        num_epoch_no_improvement += 1


    if num_epoch_no_improvement == 40:
        print("Early Stopping")
        break
    sys.stdout.flush()






























































































