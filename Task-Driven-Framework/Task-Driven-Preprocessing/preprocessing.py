from fileinput import filename
import pydicom
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from monai.transforms import SpatialPad
from scipy.ndimage import binary_erosion

def window_image(img, window_center,window_width, intercept, slope, rescale=True):
    img = (img*slope +intercept) #for translation adjustments given in the dicom file.
    img_min = window_center - window_width // 2  # minimum HU level
    img_max = window_center + window_width // 2  # maximum HU level
    img[img < img_min] = img_min  # set img_min for all HU levels less than minimum HU level
    img[img > img_max] = img_max  # set img_max for all HU levels higher than maximum HU level
    return img

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

volumi_falliti = []

path = r""
path_seg=r""
path_o=r""
path_o1=r""
size= []


# at first, it's crucial to read dicom files
cartelle = [nome for nome in os.listdir(path) if os.path.isdir(os.path.join(path, nome))]
numero_cartelle = len(cartelle)
size=[ ]
slice_thick=[]
for j in range(numero_cartelle):
    dicom_folder=path+'/'+cartelle[j]

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

    #-----------------------------------------------------------------------------------------------------------------

    mask_path = path_seg + '/' + cartelle[j] + '/' + 'lung.nii.gz'
    mask = sitk.ReadImage(mask_path)
    mask_np = sitk.GetArrayFromImage(mask)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)  # Imposta il volume come riferimento
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Usa l'interpolazione più adatta per le maschere
    resampler.SetDefaultPixelValue(0)  # Imposta il valore di default fuori dalla maschera
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))  # Trasformazione identica

    mask = resampler.Execute(mask)
    mask_np = sitk.GetArrayFromImage(mask)

    spacing_z=int(round(original_size[2] * (original_spacing[2] / target_spacing[0])))
    spacing_x = int(round(original_size[0] * (original_spacing[0] / target_spacing[1])))
    spacing_y= int(round(original_size[1] * (original_spacing[1] / target_spacing[2])))
    new_size = [spacing_z,spacing_y,spacing_x]
    size.append(new_size)
    print(new_size)
    print(img3d.shape)
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
        print(f"❌ Error performing the removal of the region outside lungs {cartelle[j]}: {e}")
        volumi_falliti.append(f"{cartelle[j]} - Error : {str(e)}")
        continue


    image_finale = sitk.GetImageFromArray(image)
    image_finale.SetSpacing(ref.GetSpacing())
    image_finale.SetOrigin(ref.GetOrigin())
    image_finale.SetDirection(ref.GetDirection())
    sze=image_finale.GetSize()
    size.append(sze)


    output_path1 = path_o + "/"+ cartelle[j]+'_'+'final_volume.nii'
    sitk.WriteImage(image_finale, output_path1, useCompression=False)
    print(f"percentuale di completamento: {round(j/numero_cartelle*100,2)}%")




