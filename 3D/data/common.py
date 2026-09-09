import torch
import os
import SimpleITK as sitk 
import pickle
# import pydicom 
from tqdm import tqdm
import numpy as np
import datetime
import pandas as pd
import json 
import random 
from PIL import Image
from einops import rearrange  
import pdb
import torchvision.transforms as transforms 
import itertools 
from torch.nn import functional as F 
def mkdir(p, is_file=False):
    if is_file:
        p, _ =  os.path.split(p)
    isExists = os.path.exists(p)
    if isExists:
        pass
    else:
        os.makedirs(p)
        print("make directory successfully:{}".format(p)) 

def rgb2ycbcr(img): 
    img = img/255.
    out_img = np.matmul(
            img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [16, 128, 128]
    return out_img

def ycbcr2rgb(img): 
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836] 
    return np.clip(out_img.round(), 0, 255)

class dataIO:
    
    def __init__(self):
        self.reader = {
            '.img':self.load_itk,
            '.gz':self.load_itk, 
            '.nii':self.load_itk,
            '.bin':self.load_bin, 
            '.txt':self.load_txt, 
            '.json':self.load_json, 
            '.png':self.load_pil, 
            '.jpg':self.load_pil, 
            '.bmp':self.load_pil, 
            '.tif':self.load_pil, 
            
            }
        self.writer = {
            '.img':self.save_itk, 
            '.gz':self.save_itk, 
            '.nii':self.save_itk,
            '.bin':self.save_bin,
            '.csv':self.save_csv,
            '.txt':self.save_txt, 
            '.json':self.save_json,
            '.png':self.save_pil, 
            '.jpg':self.save_pil, 
            '.bmp':self.save_pil, 
            '.tif':self.save_pil, 
            } 
        self.data_type = {
            "CT": np.int16, 
            "PET": np.float32, 
            "MRI": np.uint16, 
            "OCT": np.uint8,
            "Pathology": np.uint8,
            "Ultrasound": np.uint8,
            "X-ray": np.uint8,
            }

    def save_bin(self, data): 
        image = data['save_data'] 
        path = data['save_path']
        with open(path, "wb") as f:
            pickle.dump(image, f) 
    def load_bin(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        result = {
            'data':data, 
            
            } 
        return result
    
    
    def save_itk(self, data): 
        image = data['save_data'] 
        path = data['save_path'] 
        spacing = data['spacing'] 
        modality = data['modality'] 
        if modality != 'PET':
            image = image.round()
        # pdb.set_trace()
        image = sitk.GetImageFromArray(image.astype(self.data_type[modality])) 
        image.SetSpacing(spacing)
        sitk.WriteImage(image, path)
        
    def load_itk(self,path): 
        data = sitk.ReadImage(path) 
        image = sitk.GetArrayFromImage(data) 
        spacing = data.GetSpacing() 
        
        result = {
            'data': image, 
            'spacing': spacing, 
            }
        return result
        

    def load_txt(self, path):
        with open(path, "r") as f:
            data = f.read() 
        result = {
            'data':data, 
            
            } 
        return result 
    
    def save_txt(self, data): 
        text = data['save_data'] 
        path = data['save_path']
        with open(path,'w') as f:
            f.write(text) 


    def load_json(self, path):
        with open(path, encoding='utf8') as f:
            data = json.load(f) 
        result = {
            'data':data, 
            
            } 
        return result 
    def save_json(self, data, path): 
        text = data['save_data'] 
        path = data['save_path']
        with open(path, "w", encoding='utf8') as f:
            json.dump(text, f, ensure_ascii=False, indent=2) 
            
    def load_pil(self, path, use_y=True):
        img = Image.open(path) 
        mode = img.mode 
        
        if mode == 'RGB' and use_y:

            ycbcr_img = rgb2ycbcr(np.array(img))

            y = ycbcr_img[:, :, 0] 
            cbcr = ycbcr_img[:, :, -2:] 
            
            result = { 
                'data': y, 
                'mode': mode, 
                'cbcr': cbcr
                } 
        else: 
            # print("gray mode")
            img = np.array(img) 
            result = { 
                'data': img, 
                'mode': mode, 
                'cbcr': -1
                } 
            
        return result 
    
    def save_pil(self, data, use_y=True): 
        mode = data['mode'] 
        if mode=='RGB' and use_y: 
            # pdb.set_trace() 
            y = data['save_data'] 
            y = np.expand_dims(y, axis=-1)
            y_cbcr = np.concatenate([y, data['cbcr']], axis=-1) 
            # pdb.set_trace() 
            img = ycbcr2rgb(y_cbcr)
            img = Image.fromarray(img.round().astype(self.data_type[data['modality']]), mode)
        else:   
            img = Image.fromarray(data['save_data'].round().astype(self.data_type[data['modality']]), mode)
        img.save(data['save_path'], mode=mode)
        
        
        
    def save_csv(self, data): 
        data_dict = data['save_data'] 
        path = data['save_path']
        result=pd.DataFrame({ key:pd.Series(value) for key, value in data_dict.items() })
        result.to_csv(path)
            

        
    def getFileEX(self, s):
        _, tempfilename = os.path.split(s)
        _, ex = os.path.splitext(tempfilename)
        return ex
    
    def load(self, path):
        ex = self.getFileEX(path)
        return self.reader[ex](path)
    def save(self, data): 
        path = data['save_path']
        mkdir(path, is_file=True)
        ex = self.getFileEX(path)
        return self.writer[ex](data)



class transformData: 
    
    '''
    all-in-one medical image data
    '''
    
    def __init__(self):
        self.data_range = { 
            "CT": [-1024.0, 3071.0], 
            "PET": [0.0, 20.0], 
            "MRI": [0.0, 4095.0], 
            "OCT": [0.0, 255.0],
            "Pathology": [0.0, 255.0],
            "Ultrasound": [0.0, 255.0],
            "X-ray": [0.0, 255.0],
            } 



    def truncate(self,img, d_min, d_max):
        img[img>d_max]=d_max
        img[img<d_min]=d_min
        return img 
    
    def normalize(self, img, modality): 
        d_min, d_max = self.data_range[modality] 
        img = self.truncate(img, d_min, d_max) 
        img = torch.divide(img - d_min, d_max - d_min)
        return img 
    
    def denormalize(self, img, modality): 
        d_min, d_max = self.data_range[modality] 
        img = img*(d_max - d_min) + d_min 
        img = self.truncate(img, d_min, d_max)
        return img 
        
    
    def random_crop_and_augment(self, lq_img, hq_img, patch_size):
        # 确保输入是 [H, W] 的 tensor
        assert lq_img.shape == hq_img.shape, "lq_img and hq_img must have the same shape."
        
        H, W = lq_img.shape
        if patch_size > H or patch_size > W:
            raise ValueError("patch_size must be less than or equal to the dimensions of the images.")
        
        # 随机裁剪的起始位置
        top = random.randint(0, H - patch_size)
        left = random.randint(0, W - patch_size)
    
        # 裁剪
        lq_patch = lq_img[top:top + patch_size, left:left + patch_size]
        hq_patch = hq_img[top:top + patch_size, left:left + patch_size]
    
        # 随机旋转
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            lq_patch = transforms.functional.rotate(lq_patch.unsqueeze(0), angle=angle).squeeze(0)
            hq_patch = transforms.functional.rotate(hq_patch.unsqueeze(0), angle=angle).squeeze(0)
    
        # 随机翻转
        if random.random() < 0.5:
            lq_patch = transforms.functional.hflip(lq_patch.unsqueeze(0)).squeeze(0)
            hq_patch = transforms.functional.hflip(hq_patch.unsqueeze(0)).squeeze(0)
        
        if random.random() < 0.5:
            lq_patch = transforms.functional.vflip(lq_patch.unsqueeze(0)).squeeze(0)
            hq_patch = transforms.functional.vflip(hq_patch.unsqueeze(0)).squeeze(0)
    
        return lq_patch, hq_patch 

    def random_crop_and_augment_3D(self,
                                lq_img,
                                hq_img,
                                patch_size,
                                allow_reflection = True,
                                allow_axis_swap = True):
        assert isinstance(lq_img, torch.Tensor) and isinstance(hq_img, torch.Tensor), "inputs must be torch.Tensor"
        assert lq_img.shape == hq_img.shape, "lq_img and hq_img must have the same shape."
        assert lq_img.ndim == 3, "Expect 3D tensors with shape [D, H, W]."
    
        D, H, W = lq_img.shape
    
        # 规范化 patch_size
        if isinstance(patch_size, int):
            pd = ph = pw = patch_size
        else:
            if len(patch_size) != 3:
                raise ValueError("patch_size must be int or tuple of three ints (pd, ph, pw).")
            pd, ph, pw = patch_size
    
        if pd > D or ph > H or pw > W:
            raise ValueError("patch_size must be less than or equal to the corresponding image dimensions.")
    
        # 随机裁剪起始位置
        d0 = random.randint(0, D - pd) if D - pd > 0 else 0
        t0 = random.randint(0, H - ph) if H - ph > 0 else 0
        l0 = random.randint(0, W - pw) if W - pw > 0 else 0
    
        lq_patch = lq_img[d0:d0 + pd, t0:t0 + ph, l0:l0 + pw].clone()
        hq_patch = hq_img[d0:d0 + pd, t0:t0 + ph, l0:l0 + pw].clone()
    
        # 所有轴的 permutation（6 种）
        perms = list(itertools.permutations((0, 1, 2)))  # tuples of axis indices
    
        # 如果不允许轴换位，则筛选出那些不会改变 shape 的 permutation
        orig_sizes = (pd, ph, pw)
        if not allow_axis_swap:
            valid_perms = []
            for p in perms:
                permuted_sizes = (orig_sizes[p[0]], orig_sizes[p[1]], orig_sizes[p[2]])
                if permuted_sizes == orig_sizes:
                    valid_perms.append(p)
            # 如果没有任何 permutation 保持 shape（常见于非立方体），就只用 identity
            if not valid_perms:
                valid_perms = [(0, 1, 2)]
            perms = valid_perms

        perm = random.choice(perms)
    
        if allow_reflection:
            flips = [random.random() < 0.5 for _ in range(3)]
        else:
            flips = [False, False, False]

        if perm != (0, 1, 2):
            lq_patch = lq_patch.permute(perm)
            hq_patch = hq_patch.permute(perm) 
            
        flip_dims = tuple(i for i, f in enumerate(flips) if f)
        if flip_dims:
            lq_patch = torch.flip(lq_patch, dims=flip_dims)
            hq_patch = torch.flip(hq_patch, dims=flip_dims)
    
        return lq_patch, hq_patch
    
    def preprocess(self, data): 
       
        lq_img = data['lq_img'] 
        hq_img = data['hq_img'] 
        modality = data['modality'] 
        patch_size = data['patch_size']
        
        lq_img = torch.from_numpy(lq_img) 
        hq_img = torch.from_numpy(hq_img) 
        

        
        lq_img = self.normalize(lq_img, modality) 
        hq_img = self.normalize(hq_img, modality) 
        
        if patch_size!=-1: 
            
            D, H, W = lq_img.shape
            if patch_size > D or patch_size > H or patch_size > W: 
                print(lq_img.shape) 
                print(patch_size)
                print(data['lq_path'])
            
            lq_img, hq_img = self.random_crop_and_augment_3D(lq_img, hq_img, patch_size)

        lq_img = lq_img.unsqueeze(0) # [C, H, W] 
        hq_img = hq_img.unsqueeze(0) # [C, H, W]  
        
        data['lq_img'] = lq_img 
        data['hq_img'] = hq_img 
        return data 
    
    def postprocess(self, data, process_hq = False): 

        for i in range(len(data)):
            # pdb.set_trace()
            modality = data[i]['modality'] 
            save_img = data[i]['save_data'] 
            save_img = self.denormalize(save_img, modality)  
            save_img = save_img.numpy().squeeze()
            # if modality != 'PET': 
            #     save_img = save_img.round().astype(self.data_type[modality])
            
            if process_hq:
                hq_img = data[i]['hq_img'] 
                hq_img = self.denormalize(hq_img, modality)  
                hq_img = hq_img.numpy().squeeze()
                # if modality != 'PET':
                #     hq_img = hq_img.round().astype(self.data_type[modality])
                data[i]['hq_img'] = hq_img
            
            data[i]['save_data'] = save_img 
            
        return data

@torch.no_grad()
def restore_with_patches(model, img, kernel_size=64, stride=32, crop_size = 3, batch_compute=False): 
    # img = input_img.clone()
    B, C, D, H, W = img.shape
    # Check if H and W are greater than kernel_size
    # if D < kernel_size or H < kernel_size or W < kernel_size:
    #     raise ValueError("D, H and W must be greater than {}".format(kernel_size)) 
    
    pad_d = (stride - D % stride) if D % stride != 0 else 0
    pad_h = (stride - H % stride) if H % stride != 0 else 0
    pad_w = (stride - W % stride) if W % stride != 0 else 0 


    # If padding is needed, calculate symmetric padding
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # Symmetric padding for depth (front, back)
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front
        # Symmetric padding for height (top, bottom)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        # Symmetric padding for width (left, right)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding
        padding = (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)  # (left, right, top, bottom, front, back)
        img = F.pad(img, padding, mode='constant', value=0) 
    else: 
        padding = (0, 0, 0, 0, 0, 0) 
    

    
    # Process input with Model
    B, C, D, H, W = img.shape
    nz = int(D//stride)-1
    nx = int(H//stride)-1
    ny = int(W//stride)-1
    result = torch.zeros((D, H, W)).type(torch.FloatTensor).to(img.device)
    flag=True
    for k in tqdm(list(range(nz))):
        idz = 0 if k==0 else k*stride+kernel_size-stride-crop_size
        if idz+crop_size+stride>D:
            flag=False
        x = img[:,:,k*stride:k*stride+kernel_size,:,:] if flag else img[:,:,D-kernel_size:,:,:]##Large patches along z axis
        patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride).unfold(4, kernel_size, stride)
        patches=patches.reshape(-1, C, kernel_size, kernel_size, kernel_size)
        ######
        #Synthesis
        ###### 
        
        G_patches = patches.clone().detach() 
        if batch_compute: 
            G_patches = model(patches) 
        else:
            for i in range(len(patches)):
                G_patches[i] = model(patches[[i],:,:,:,:]) 
                
        for i in range(nx):
            idx = 0 if i==0 else i*stride+kernel_size-stride-crop_size
            for j in range(ny):
                idy = 0 if j==0 else j*stride+kernel_size-stride-crop_size
                if flag:
                    result[idz:k*stride+kernel_size,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*ny+j, 0,idz-k*stride:,idx-i*stride:,idy-j*stride:]
                else:
                    result[idz:,idx:i*stride+kernel_size, idy:j*stride+kernel_size] = G_patches[i*ny+j, 0,idz+kernel_size-D:,idx-i*stride:,idy-j*stride:] 

    # Remove padding 
    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = padding 
    
    # import pdb 
    # pdb.set_trace()
    # result = result[pad_front:-pad_back, pad_top:-pad_bottom, pad_left:-pad_right]
    if pad_back > 0:
        result = result[:-pad_back, :, :]
    if pad_front > 0:
        result = result[pad_front:, :, :]
    if pad_bottom > 0:
        result = result[:, :-pad_bottom, :]
    if pad_top > 0:
        result = result[:, pad_top:, :]
    if pad_right > 0:
        result = result[:, :, :-pad_right]
    if pad_left > 0:
        result = result[:, :, pad_left:] 
        

    return result.unsqueeze(0).unsqueeze(0)