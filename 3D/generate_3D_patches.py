import torch
import torch.nn as nn
import pickle
import os 
import numpy as np 
import SimpleITK as sitk  
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
def calculate_variance(data):
    '''
    input shape: [N, C, D, H, W]
    var(x) = E[x^2] - E[x]^2
    '''
    avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
    E_x2 = avg_pool(data**2)
    Ex_2 = avg_pool(data)**2
    
    return E_x2 - Ex_2 

def save_itk(image, path, spacing=None): 
    image = sitk.GetImageFromArray(image) 
    if spacing is not None:
        image.SetSpacing(spacing)
    sitk.WriteImage(image, path)
    
def load_itk(path, return_spacing=False): 
    data = sitk.ReadImage(path) 
    image = sitk.GetArrayFromImage(data) 
    spacing = data.GetSpacing() 
    if return_spacing:
        return image, spacing 
    else:
        return image 

def binary(data, thresh): 
    result = np.copy(data)
    result[data>=thresh]=1
    result[data<thresh]=0 
    return result

def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_pkl(data, path):
    with open(path, "wb") as f:
        data = pickle.dump(data, f)
    return data

def mkdir(p, is_file=False):
    if is_file:
        p, _ =  os.path.split(p)
    isExists = os.path.exists(p)
    if isExists:
        pass
    else:
        os.makedirs(p)
        print("make directory successfully:{}".format(p)) 

def unfold_3d_image(image, block_size=(64, 64, 64), overlap=32):
    D, H, W = image.shape
    block_d, block_h, block_w = block_size
    
    step = block_size[0] - overlap  
    N_d = (D - overlap) // step  
    N_h = (H - overlap) // step  
    N_w = (W - overlap) // step  
    
    N = N_d * N_h * N_w
    
    unfolded = np.zeros((N, block_d, block_h, block_w), dtype=image.dtype)
    
    block_idx = 0
    for i in range(N_d):
        for j in range(N_h):
            for k in range(N_w):
                start_d = i * step
                start_h = j * step
                start_w = k * step

                unfolded[block_idx] = image[start_d:start_d+block_d, start_h:start_h+block_h, start_w:start_w+block_w]
                block_idx += 1
                
    return unfolded

root = "/home/data/zhiwen/dataset/All-in-One-7-Task/3D/"
modality_list = ["PET", "CT", "MRI"] 
thresh_list = {
    'PET': 0.05, 
    'CT': -800, 
    'MRI':100
    }
patch_size = 64 # 64*64*64

def save_patches(root, name, modality, thresh, patch_size, overlap):
    hq = load_itk(os.path.join(root, modality, "train", "HQ", name)) 
    lq = load_itk(os.path.join(root, modality, "train", "LQ", name)) 
    mask = binary(hq, thresh) 
    
    
    hq = unfold_3d_image(hq, block_size=(patch_size, patch_size, patch_size), overlap=overlap) 
    lq = unfold_3d_image(lq, block_size=(patch_size, patch_size, patch_size), overlap=overlap) 
    mask = unfold_3d_image(mask, block_size=(patch_size, patch_size, patch_size), overlap=overlap) 
    
    assert hq.shape == lq.shape and hq.shape == mask.shape
    for k in range(hq.shape[0]): 
        if np.sum(mask[k]) / patch_size**3 > 0.3: 
            save_dir = os.path.join(root, modality, "train_patch") 
            mkdir(os.path.join(save_dir, "HQ")) 
            mkdir(os.path.join(save_dir, "LQ"))

            save_itk(hq[k], os.path.join(save_dir, "HQ", name.replace(".nii.gz", "_{}.nii.gz".format(k)))) 
            save_itk(lq[k], os.path.join(save_dir, "LQ", name.replace(".nii.gz", "_{}.nii.gz".format(k)))) 


with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [] 
    for modality in modality_list:
        name_list = os.listdir(os.path.join(root, modality, "train", "HQ")) 
        for name in name_list:
            futures.append(executor.submit(save_patches, root, name, modality, thresh_list[modality], patch_size, patch_size//4))
        

    for future in tqdm(as_completed(futures)):
        future.result()  
# save_patches(root, modality, thresh_list[modality], patch_size)
    
# # data = torch.from_numpy(data).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0) 
# # variance = calculate_variance(data) 

# mask = binary(data, thresh) 
# mask = mask.astype(np.uint8)
    


