import torch
from torch.utils.data import Dataset
import os
import numpy as np
from .common import dataIO, transformData
import pdb 
import random
io=dataIO() 
transform = transformData()

class MedIRData(Dataset):
    def __init__(self, root_dir, modality_list = ["PET"], patch_size=64, preprocess=True, flag='train', save_root=-1, use_num=-1):
        
        self.paired_info = []
        
        for modality in modality_list:
            
            file_list = os.listdir(os.path.join(root_dir, modality, flag, 'HQ'))
            
            use_num_tmp = len(file_list) if use_num == -1 else use_num
            
            for i in range(use_num_tmp):  
                tmp = { 
                    'lq_path': os.path.join(root_dir, modality, flag, 'LQ', file_list[i]), 
                    'hq_path': os.path.join(root_dir, modality, flag, 'HQ', file_list[i]), 
                    'modality': modality, 
                    'file_name': file_list[i]
                    } 
                self.paired_info.append(tmp)

        self.length = len(self.paired_info) 
        self.patch_size = patch_size 
        self.preprocess = preprocess 
        self.save_root = save_root ## for test


    def __len__(self):
        return self.length 



    def __getitem__(self, idx): 
        

        
        info = self.paired_info[idx] 
        lq_path = info['lq_path'] 
        hq_path = info['hq_path'] 
        modality = info['modality'] 
        file_name = info['file_name'] 
        


       
        lq_data = io.load(lq_path) 
        hq_data = io.load(hq_path) 
        
        lq_img = lq_data['data'] 
        hq_img = hq_data['data'] 
        

        
        if modality in ['PET', 'CT', 'MRI']:
            spacing = np.array(lq_data['spacing']) 
        else:
            spacing = np.array([1, 1, 1]) 
        
            
        
        if modality == 'Pathology':
            cbcr = lq_data['cbcr'] 
            mode = lq_data['mode'] 
        elif modality in ["OCT", "Ultrasound", "X-ray"]:
            cbcr = -1 
            mode = lq_data['mode'] 
        else:
            cbcr = -1 
            mode = -1 
        
        # pdb.set_trace()
        
        if self.save_root!=-1:
            save_path = os.path.join(self.save_root, modality, file_name) 
        else:
            save_path = -1
        
        
        

        data_dict = {
            'lq_img': lq_img,
            'hq_img': hq_img,
            'save_data': -1,
            'cbcr': cbcr,   ## cbcr of lq image
            'lq_path': lq_path,
            'hq_path': hq_path,
            'save_path': save_path,
            'modality': modality, 
            'file_name': file_name, 
            'spacing': spacing,
            'mode': mode, 
            'patch_size': self.patch_size
            } 
        

        
        if self.preprocess:
            data_dict = transform.preprocess(data_dict) 
        return data_dict




class DataSampler:
    def __init__(self, dataloader_list):
        self.dataloader_list = dataloader_list
        # 为每个 dataloader 创建一个迭代器列表
        self.data_iters = [iter(dataloader) for dataloader in dataloader_list]

    def __iter__(self):
        return self

    def __next__(self):
        batches = [] 
        # 遍历每个 dataloader 迭代器
        for i, data_iter in enumerate(self.data_iters):
            try: 
                # pdb.set_trace()
                # 从当前迭代器中获取下一个批次
                batch = next(data_iter)
            except StopIteration:
                # 如果某个 dataloader 采样完了，重新生成一个新的迭代器
                self.data_iters[i] = iter(self.dataloader_list[i])
                batch = next(self.data_iters[i])
            # 将采样结果加入到 batches 列表中
            batches = batches + batch 
        
        random.shuffle(batches)
        
        return batches






def custom_collate_fn(batch):
    # 初始化一个空列表用于存放新的batch
    new_batch = []
    
    for item in batch:
        # 将每个item初始化为空字典
        new_item = {}
        
        for key, value in item.items():
            if key in ['lq_img', 'hq_img']:
                # 如果键是 'lq_img' 或 'hq_img'，将其值转换为 tensor
                new_item[key] = value.unsqueeze(0)
            else:
                # 否则，保持原样
                new_item[key] = value
        
        new_batch.append(new_item)
    
    return new_batch









# dataset = Train_Data() 
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True,drop_last=True) 
# data_sampler = DataSampler(data_loader) 

if __name__ == "__main__": 
    from tqdm import tqdm 
    from torch.utils.data import DataLoader
    
    data_root = "/home/data/zhiwen/dataset/All-in-One-7-Task/3D/" 
    # modality_list = ["PET", "CT", "MRI", "OCT", "Pathology", "Ultrasound", "X-ray"] 
    modality_list = ["PET"]
    save_root = "/home/data/zhiwen/experiment/All-in-One-7-Task/3D/data/"

    
    dataset = {
        'train': MedIRData(root_dir=data_root, modality_list = modality_list, patch_size=64, preprocess=True, flag='train_patch', save_root=save_root), 
        'test': MedIRData(root_dir=data_root, modality_list = modality_list, patch_size=-1, preprocess=True, flag='test', save_root=save_root), 
        } 
    # pdb.set_trace()
    train_loader = DataLoader(dataset['train'], batch_size=1, shuffle=False, collate_fn=custom_collate_fn) 
    print("length:", len(train_loader))

    for counter, data in enumerate(tqdm(train_loader)): 
        
        lq_img_data = [item['lq_img'] for item in data]
        hq_img_data = [item['hq_img'] for item in data]  
        
        lq_img = torch.cat(lq_img_data, dim=0)
        hq_img = torch.cat(hq_img_data, dim=0) 

