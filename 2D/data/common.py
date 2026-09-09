import torch
import os
import SimpleITK as sitk 
import pickle
# import pydicom
import numpy as np
import datetime
import pandas as pd
import json 
import random 
from PIL import Image
from einops import rearrange  
import pdb
import torchvision.transforms as transforms
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
        spacing = data.get('spacing') 
        modality = data.get('modality') 
        round_data = data.get('round') 
        if round_data:
            image = image.round()
        if modality is not None:
            image = image.astype(self.data_type[modality])
        image = sitk.GetImageFromArray(image) 
        if spacing is not None:
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
            
            H, W = lq_img.shape
            if patch_size > H or patch_size > W: 
                print(lq_img.shape) 
                print(patch_size)
                print(data['lq_path'])
            
            lq_img, hq_img = self.random_crop_and_augment(lq_img, hq_img, patch_size)

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

class ShapeAligner:
    def __init__(self):

        self.original_sizes = []

    def pad_images(self, image_list): 
        padded_images = []
        self.original_sizes = []
        max_height = max(img.shape[2] for img in image_list)
        max_width = max(img.shape[3] for img in image_list)

        for img in image_list:
            _, _, h, w = img.shape
            padding = (0, max_width - w, 0, max_height - h)
            padded_img = torch.nn.functional.pad(img, padding)
            padded_images.append(padded_img)
            self.original_sizes.append((h, w))

        return torch.cat(padded_images, dim=0)

    def unpad_images(self, padded_batch):
        unpadded_images = []
        for i, (h, w) in enumerate(self.original_sizes): 
            
            unpadded_img = padded_batch[[i], :, :h, :w]
            unpadded_images.append(unpadded_img) 
        # pdb.set_trace()
        # self.original_sizes = []
        return unpadded_images
