from fileinput import filename
import pydicom
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from monai.transforms import SpatialPad
from scipy.ndimage import binary_erosion
import os
import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator
from monai.transforms import SpatialPad, Resize
failed_files = []
import pandas as pd
import time
from scipy.ndimage import binary_erosion

def window_image(img, window_center,window_width, intercept, slope, rescale=True):
    img = (img*slope +intercept) #for translation adjustments given in the dicom file.
    img_min = window_center - window_width // 2  # minimum HU level
    img_max = window_center + window_width // 2  # maximum HU level
    img[img < img_min] = img_min  # set img_min for all HU levels less than minimum HU level
    img[img > img_max] = img_max  # set img_max for all HU levels higher than maximum HU level
    return img

def block_max(vol: torch.Tensor, S: int) -> torch.Tensor:
   
    N, H, W = vol.shape
    num_blocks = (N + S - 1) // S  # include anche l'ultimo blocco se non completo

    result = []
    for i in range(0, N, S):
        block = vol[i:i+S]             # shape (S, H, W) o meno se ultimo blocco
        result.append(block.max(dim=0).values)

    return torch.stack(result)

def lung_segmentation(dicom_folder, cartella, path_seg):
    output_path = os.path.join(path_seg, cartella)
    print(f"\n➡️ Segmentazione di: {dicom_folder}")
    try:
        totalsegmentator(dicom_folder, output_path, task="lung_nodules", device = 'gpu' )
    except Exception as e:
        print(f"Errore durante l'elaborazione di: {cartella}")
        print(f"Dettagli errore: {e}")


def resize_volume_torch(volume_np, new_shape, mode='trilinear'):
    volume = torch.from_numpy(volume_np).unsqueeze(0).unsqueeze(0).float()
    volume_resized = F.interpolate(
        volume,
        size=new_shape,
        mode=mode,
        align_corners=False if mode != 'nearest' else None
    )
    # Rimuovi batch e channel → torna a (Z, Y, X)
    volume_resized_np = volume_resized.squeeze().cpu().numpy()
    return volume_resized_np

def main():

    path = r""
    path_seg=r""
    path_o=r""

    resize = Resize(spatial_size=(240, 256, 256), mode='trilinear', align_corners=False)
 


    # at first, it's crucial to read dicom files
    cartelle = [nome for nome in os.listdir(path) if os.path.isdir(os.path.join(path, nome))]
    numero_cartelle = len(cartelle)
    
    start_time = time.perf_counter()

    for j in range(numero_cartelle):

        
        dicom_folder=path+'/'+cartelle[j]
        lung_segmentation(dicom_folder, cartelle[j],path_seg)

        files = []
        for filename in os.listdir(dicom_folder):
            filepath = os.path.join(dicom_folder, filename)

            if filename!='VERSION':
                files.append(pydicom.dcmread(filepath))

        slices = []
        skipcount = 0
        for f in files:
            if hasattr(f, "SliceLocation"):
                slices.append(f)
            else:
                skipcount = skipcount + 1

        # ensure they are in the correct order
        slices = sorted(slices, key=lambda s: s.SliceLocation)

        # create 3D array
        img_shape = list(slices[0].pixel_array.shape)
        img_shape.insert(0,len(slices))
        img3d = np.zeros(img_shape)

        # fill 3D array with the images from the files
        for i, s in enumerate(slices):
            img2d = s.pixel_array
            img3d[i, :, :] = img2d


        dicom_fields = [slices[0][('0028','1050')].value, #window center
                        slices[0][('0028','1051')].value, #window width
                        slices[0][('0028','1052')].value, #intercept
                        slices[0][('0028','1053')].value] #slope

        slice_thickness = float(slices[0].SliceThickness)  # (0018,0050) # spessore della fetta

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
        reader.SetFileNames(dicom_names)
        ref = reader.Execute()
        original_spacing = ref.GetSpacing()
        original_size = ref.GetSize()
        target_spacing = [1 ,0.673828125, 0.673828125]
        mask_path = path_seg + '/' + cartelle[j] + '/' + 'lung.nii.gz'
        mask = sitk.ReadImage(mask_path)
        mask_np = sitk.GetArrayFromImage(mask)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)  
        resampler.SetInterpolator(sitk.sitkNearestNeighbor) 
        resampler.SetDefaultPixelValue(0)  
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity)) 

        mask = resampler.Execute(mask)
        mask_np = sitk.GetArrayFromImage(mask)

        spacing_z=int(round(original_size[2] * (original_spacing[2] / target_spacing[0])))
        spacing_x = int(round(original_size[0] * (original_spacing[0] / target_spacing[1])))
        spacing_y= int(round(original_size[1] * (original_spacing[1] / target_spacing[2])))
        new_size = [spacing_z,spacing_y,spacing_x]
 
        img3d=resize_volume_torch(img3d, new_size, mode='trilinear')
        mask_np=resize_volume_torch(mask_np, new_size, mode='nearest')

        # we are making use of a bounding box (to underline the region in which lungs are placed)


        image = window_image(img3d, dicom_fields[0], dicom_fields[1], dicom_fields[2], dicom_fields[3], rescale=True)
        try:
            minimo=np.min(image[mask_np>0])
            massimo=np.max(image[mask_np>0])


            image = (image-minimo)/(massimo-minimo)
            mask_np = binary_erosion(mask_np, iterations=12)

            image = image * mask_np
            coords = np.argwhere(mask_np > 0)

            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)
            
            padding=20
            z_min_p = max(z_min-padding, 0)
            y_min_p = max(y_min - padding, 0)
            x_min_p = max(x_min - padding, 0)

            z_max_p = min(z_max + padding, mask_np.shape[0] - 1)
            y_max_p = min(y_max + padding, mask_np.shape[1] - 1)
            x_max_p = min(x_max + padding, mask_np.shape[2] - 1)

            image = image[z_min:z_max+1, y_min_p:y_max_p+1, x_min_p:x_max_p+1]


        except Exception as e:
            print(f"❌ Error {cartelle[j]}: {e}")
            continue



 
        CurrentVol = torch.from_numpy(image)
        CurrentVol = CurrentVol.unsqueeze(0)
        CurrentVol = resize(CurrentVol)
        vol = CurrentVol[0, :, :, :]
        vol_finale = block_max(vol, 10)    


        image_finale = sitk.GetImageFromArray(vol_finale)
        image_finale.SetSpacing(ref.GetSpacing())
        image_finale.SetOrigin(ref.GetOrigin())
        image_finale.SetDirection(ref.GetDirection())
        sze=image_finale.GetSize()


        output_path1 = path_o + "/"+ cartelle[j]+'_'+'final_volume.nii'


        sitk.WriteImage(image_finale, output_path1, useCompression=False)
        print(f"percentage: {round(j/numero_cartelle*100,2)}%")

    end_time = time.perf_counter()

    tempo = end_time-start_time
    print(tempo)


if __name__ == "__main__":
    main()

