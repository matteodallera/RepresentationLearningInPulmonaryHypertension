import os
import numpy as np
import SimpleITK as sitk
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
from pathlib import Path
import pydicom
import numpy as np
import SimpleITK as sitk
import numpy as np
import SimpleITK as sitk



def numpy_to_sitk(volume_zyx: np.ndarray, spacing_xyz):
    img = sitk.GetImageFromArray(volume_zyx)  # volume è (Z,Y,X)
    sx, sy, sz = spacing_xyz
    img.SetSpacing((sx, sy, sz))  # SITK vuole (X,Y,Z)
    return img



def read_dicom_volume_pydicom(dicom_dir: str):

    dicom_dir = Path(dicom_dir)

    # 1) lista file (anche nested? se vuoi rglob)
    files = [p for p in dicom_dir.iterdir() if p.is_file() and p.name != "VERSION"]
    if not files:
        raise RuntimeError(f"Nessun file trovato in {dicom_dir}")

    slices = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), force=True)
            if "PixelData" not in ds:
                continue  # non-image DICOM
            slices.append(ds)
        except Exception:
            continue

    dicom_fields = [slices[0][('0028','1050')].value, #window center
                    slices[0][('0028','1051')].value, #window width
                    slices[0][('0028','1052')].value, #intercept
                    slices[0][('0028','1053')].value,
                    slices[0][('0018','0060')].value,
                    slices[0][('0018','1210')].value
                    ] #slope

    if not slices:
        raise RuntimeError(f"Nessun DICOM immagine (PixelData) in {dicom_dir}")

    # 2) ordinamento robusto
    def sort_key(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            return float(ipp[2])
        inst = getattr(ds, "InstanceNumber", None)
        if inst is not None:
            return float(inst)
        # fallback su SliceLocation (se c'è)
        sl = getattr(ds, "SliceLocation", None)
        if sl is not None:
            return float(sl)
        return 0.0

    slices.sort(key=sort_key)

    # 3) stack volume (Z,Y,X)
    vol = np.stack([ds.pixel_array for ds in slices], axis=0).astype(np.float32)



    # 5) spacing
    ps = getattr(slices[0], "PixelSpacing", None)
    if ps is None:
        raise RuntimeError("PixelSpacing mancante (0028,0030)")
    sy, sx = float(ps[0]), float(ps[1])

    # z spacing: preferisci differenze IPP
    z_positions = []
    for ds in slices:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            z_positions.append(float(ipp[2]))

    if len(z_positions) >= 2:
        z_positions = np.array(z_positions, dtype=np.float32)
        sz = float(np.median(np.abs(np.diff(z_positions))))
    else:
        # fallback
        sz = float(getattr(slices[0], "SpacingBetweenSlices",
                           getattr(slices[0], "SliceThickness", 1.0)))

    return vol, (sx, sy, sz), dicom_fields




def window_image(img, window_center,window_width, intercept, slope, rescale=True):
    img = (img*slope +intercept) #for translation adjustments given in the dicom file.
    img_min = window_center - window_width // 2  # minimum HU level
    img_max = window_center + window_width // 2  # maximum HU level
    img[img < img_min] = img_min  # set img_min for all HU levels less than minimum HU level
    img[img > img_max] = img_max  # set img_max for all HU levels higher than maximum HU level
    return img

def resize_volume_torch(volume_np, new_shape, mode='trilinear'):
    volume = torch.from_numpy(volume_np.copy()).unsqueeze(0).unsqueeze(0).float()
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

size= []


# at first, it's crucial to read dicom files
cartelle = [nome for nome in os.listdir(path) if os.path.isdir(os.path.join(path, nome))]
numero_cartelle = len(cartelle)
size=[ ]
slice_thick=[]
kvs = [ ]
kernels = [ ]
for j in range(numero_cartelle):
    print(cartelle[j])
    dicom_folder = os.path.join(path, cartelle[j])
    vol, spacing, dicom_fields = read_dicom_volume_pydicom(dicom_folder)

    kvs.append(dicom_fields[4])
    kernels.append(dicom_fields[5])

    ref = numpy_to_sitk(vol, spacing)
    original_spacing = ref.GetSpacing()
    slice_thick.append(original_spacing)
    original_size = ref.GetSize()
    target_spacing = [1.25 ,0.673828125, 0.673828125]
    #-----------------------------------------------------------------------------------------------------------------


    mask_path = path_seg + '/' + cartelle[j] + '/' + 'lung.nii.gz'
    mask = sitk.ReadImage(mask_path)
    mask_np = sitk.GetArrayFromImage(mask)
    mask_np = np.flip(mask_np, axis=1)
    #resampler = sitk.ResampleImageFilter()
    #resampler.SetReferenceImage(ref)  # Imposta il volume come riferimento
    #resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Usa l'interpolazione più adatta per le maschere
    #resampler.SetDefaultPixelValue(0)  # Imposta il valore di default fuori dalla maschera
    #resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))  # Trasformazione identica

    #mask = resampler.Execute(mask)
    #mask_np = sitk.GetArrayFromImage(mask)

    spacing_z=int(round(original_size[2] * (original_spacing[2] / target_spacing[0])))
    spacing_x = int(round(original_size[0] * (original_spacing[0] / target_spacing[1])))
    spacing_y= int(round(original_size[1] * (original_spacing[1] / target_spacing[2])))
    new_size = [spacing_z,spacing_y,spacing_x]
    size.append(new_size)
    img3d = vol
    img3d=resize_volume_torch(img3d, new_size, mode='trilinear')
    mask_np=resize_volume_torch(mask_np, new_size, mode='nearest')


    # we are making use of a bounding box (to underline the region in which lungs are placed)
    print(dicom_fields)
    try:
        image = window_image(img3d, 40, 350, dicom_fields[2], dicom_fields[3], rescale=True)
    except Exception as e:
        print(f"❌ Error for  {cartelle[j]}: {e}")
        continue

    minimo = np.min(image)
    massimo = np.max(image)
    image = (image - minimo) / (massimo - minimo)


    image_finale = sitk.GetImageFromArray(image)
    image_finale.SetSpacing(ref.GetSpacing())
    image_finale.SetOrigin(ref.GetOrigin())
    image_finale.SetDirection(ref.GetDirection())
    sze=image_finale.GetSize()
    size.append(sze)


    output_path1 = path_o + "/"+ cartelle[j]+'_'+'new_volume.nii'


    sitk.WriteImage(image_finale, output_path1, useCompression=False)
    print(f"percentage: {round(j/numero_cartelle*100,2)}%")




