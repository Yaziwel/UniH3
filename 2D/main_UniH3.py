import os 
os.environ['CUDA_VISIBLE_DEVICES']='1'  
from model.UniH3 import UniH3
from loss.losses import CharbonnierLoss
from evaluation.evaluation_metric import compute_measure
from data.common import transformData, dataIO, ShapeAligner
from data.MedIRData import MedIRData, DataSampler, custom_collate_fn
import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR
import time 
from tqdm import tqdm
import random
from utils.tools import set_seeds, mkdir  
import pdb 
import pandas as pd 
from ema_pytorch import EMA 
from utils.muon import SingleDeviceMuonWithAuxAdam

transform = transformData()
io=dataIO() 
set_seeds(42) 
shape_aligner = ShapeAligner()



def save_model(G_net_model, save_root, optimizer_G=None, ex=""):
    save_path=os.path.join(save_root, "Model")
    mkdir(save_path)
    G_save_path = os.path.join(save_path,'Generator{}.pth'.format(ex))
    torch.save(G_net_model.cpu().state_dict(), G_save_path)
    G_net_model.cuda()

    if optimizer_G is not None:
        opt_G_save_path = os.path.join(save_path,'Optimizer_G{}.pth'.format(ex))
        torch.save(optimizer_G.state_dict(), opt_G_save_path)



def build_train_sampler(modality_list, src_root, batch_size, save_root, sampling='uniform'): 
    dataloader_list = []
    if sampling=='uniform': 
        for modality in modality_list:
            dataset = MedIRData(root_dir=src_root, modality_list = [modality], patch_size=128, preprocess=True, flag='train', save_root=save_root, use_num=-1) 
            print("{}: {}".format(modality, dataset.length))
            dataloader_list.append(DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, collate_fn=custom_collate_fn)) 
        
    elif sampling=='random':
        dataset = MedIRData(root_dir=src_root, modality_list = modality_list, patch_size=128, preprocess=True, flag='train', save_root=save_root, use_num=-1) 
        print("All Modality: {}".format(dataset.length)) 
        dataloader_list.append(DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, collate_fn=custom_collate_fn)) 
    else:
        raise Exception("Sampling Method Error")
    sampler = DataSampler(dataloader_list) 
    return sampler



total_iteration = 1e6
val_iteration = 1e3

batch_size = 16
eps=1e-8
lr=3e-4
psnr_max=0


src_root = "/home/data/zhiwen/dataset/All-in-One-7-Task/2D/" ### Path to place data
modality_list = ["PET", "CT", "MRI", "OCT", "Pathology", "Ultrasound", "X-ray"] 
# modality_list = ["MRI"] 
save_root = "experiment/UniH3"



'''
Train
'''
print("################ Train ################")

Generator = UniH3(task_list = modality_list) 
# Generator = AE_IR(AE_ckpt_path=os.path.join(save_root, "Model", "AE_best.pth")) 
# Generator.load_state_dict(torch.load(os.path.join(save_root, "Model","Generator_best.pth"))) 
Generator.cuda() 

model_ema = EMA(
    Generator,
    beta = 0.999,              # exponential moving average factor
    update_after_step = 0,    # only after this number of .update() calls will it start updating
    update_every = 1,          # how often to actually update, to save on compute (updates every 10th .update() call) 
    # update_model_with_ema_every = 1e4,
    move_ema_to_online_device = True
)


train_sampler = build_train_sampler(modality_list, src_root, batch_size, save_root, sampling='random') 
valid_loader = DataLoader(
    MedIRData(root_dir=src_root, modality_list = modality_list, patch_size=-1, preprocess=True, flag='test', save_root=save_root, use_num=16), 
    batch_size=1, 
    shuffle=False, 
    collate_fn=custom_collate_fn
    ) 



# optimizer_G = torch.optim.AdamW(Generator.parameters(), weight_decay = 1e-4, lr=lr, betas=(0.9, 0.999))
# lr_scheduler_G = CosineAnnealingLR(optimizer_G, total_iteration, eta_min=1.0e-7) 


group_kxk = []
group_other = []
for name, param in Generator.named_parameters():
    if param.ndim == 4 and param.shape[-2:] != (1, 1):
        group_kxk.append((param))
    else:
        group_other.append((param)) 
print("length", len(group_kxk))
param_groups = [
    dict(params=group_kxk, use_muon=True,
         lr=lr*5, weight_decay=1e-4),
    dict(params=group_other, use_muon=False,
         lr=lr, betas=(0.9, 0.999), weight_decay=1e-4),
]
optimizer_G = SingleDeviceMuonWithAuxAdam(param_groups) 
lr_scheduler_G = CosineAnnealingLR(optimizer_G, total_iteration, eta_min=1.0e-7) 


running_loss = []
eval_metrics={
    "psnr":[],
    "ssim":[], 
    "rmse":[]
    } 

'''
resume
'''
resume_iter = 0
if resume_iter:
    model_ema.load_state_dict(torch.load(os.path.join(save_root, "Model","Generator_ema_best.pth"))) 
    optimizer_G.load_state_dict(torch.load(os.path.join(save_root, "Model","Optimizer_G_ema_best.pth"))) 
    eval_metrics = io.load(os.path.join(save_root, "evaluationLoss.bin"))['data'] 

pbar = tqdm(total=int(total_iteration))

for iteration in list(range(1, int(total_iteration)+1)): 
    if resume_iter>0 and pbar.n <= resume_iter: 
        lr_scheduler_G.step() 
        pbar.update() 
        continue
    l_G=[] 
    
    data = next(train_sampler)
    
    lq_img = [item['lq_img'] for item in data]
    hq_img = [item['hq_img'] for item in data] 
    m_list = [item['modality'] for item in data] 

    lq_img = shape_aligner.pad_images(lq_img)
    hq_img = shape_aligner.pad_images(hq_img) 

    lq_img = lq_img.type(torch.FloatTensor).cuda()
    hq_img = hq_img.type(torch.FloatTensor).cuda() 


    Generator.train()
    optimizer_G.zero_grad() 

    loss_G = Generator(lq_img, hq_img, m_list)  
    
    loss_G.backward()
    optimizer_G.step() 
    
    model_ema.update()
    
    # pdb.set_trace()

    l_G.append(loss_G.item())
    torch.cuda.empty_cache() 
    lr_scheduler_G.step()
    
    if iteration % val_iteration == 0: 
        psnr = []
        ssim = []
        rmse = []
        model_ema.ema_model.eval() 
        for counter, data in enumerate(tqdm(valid_loader)):

            lq_img = [item['lq_img'] for item in data]
            lq_img = shape_aligner.pad_images(lq_img)
            lq_img = lq_img.type(torch.FloatTensor).cuda()
            with torch.no_grad():
                gen_img = model_ema(lq_img).detach().cpu() 
            gen_img = shape_aligner.unpad_images(gen_img)
            
            for i in range(len(gen_img)):
                data[i]['save_data']=gen_img[i] 
            
            data = transform.postprocess(data, process_hq=True) 
            [io.save(d) for d in data] 
            
            eval_result = compute_measure(data) 
            psnr += eval_result[0]
            ssim += eval_result[1]
            rmse += eval_result[2]
            
            
            # import pdb 
            # pdb.set_trace()
            

            torch.cuda.empty_cache()
        c_psnr=np.mean(psnr)
        c_ssim=np.mean(ssim)
        c_rmse=np.mean(rmse)
        
        eval_metrics['psnr'].append(c_psnr)
        eval_metrics['ssim'].append(c_ssim)  
        eval_metrics['rmse'].append(c_rmse) 
    
        # save_model(G_net_model=Generator, save_root=save_root, optimizer_G=None, ex="_iteration_{}".format(iteration))
        if c_psnr>=psnr_max:
            psnr_max=c_psnr
            io.save(
                {"save_data": "Best Iteration: {}, PSNR: {}, SSIM:{}, RMSE:{}".format(iteration, c_psnr, c_ssim, c_rmse),
                  "save_path": os.path.join(save_root, "best.txt")
                  }
                )
            save_model(G_net_model=model_ema, save_root=save_root, optimizer_G = optimizer_G, ex="_ema_best")
        io.save(
            {'save_data': eval_metrics,
            'save_path':os.path.join(save_root, "evaluationLoss.bin")
            }
            ) 
        # print("AWL Weight:", AWL.params)



    pbar.set_description("loss_G:{:6},  psnr:{:6}".format(loss_G.item(), eval_metrics['psnr'][-1] if len(eval_metrics['psnr'])>0 else 0)) 
    pbar.update() 





'''
Test
'''
print("################ Test ################")
model_ema.load_state_dict(torch.load(os.path.join(save_root, "Model","Generator_ema_best.pth"))) 
test_loader = DataLoader(
    MedIRData(root_dir=src_root, modality_list = modality_list, patch_size=-1, preprocess=True, flag='test', save_root=save_root, use_num=-1), 
    batch_size=1, 
    shuffle=False, 
    collate_fn=custom_collate_fn
    ) 

for counter, data in enumerate(tqdm(test_loader)):

    lq_img = [item['lq_img'] for item in data]
    lq_img = shape_aligner.pad_images(lq_img)
    lq_img = lq_img.type(torch.FloatTensor).cuda()
    with torch.no_grad():
        gen_img = model_ema(lq_img).detach().cpu() 
    gen_img = shape_aligner.unpad_images(gen_img)
    
    for i in range(len(gen_img)):
        data[i]['save_data']=gen_img[i] 
    
    data = transform.postprocess(data, process_hq=True) 
    [io.save(d) for d in data] 


'''
Evaludation
'''
print("################ Evaludation ################") 

for modality in modality_list: 
    result_dict = {
        "NAME": [], 
        "PSNR": [], 
        "SSIM": [], 
        "RMSE": [],
        "PSNR_MEAN": None, 
        "SSIM_MEAN": None, 
        "RMSE_MEAN": None,
        } 
    
    hq_p = os.path.join(src_root, modality, "test", "HQ") 
    gen_p = os.path.join(save_root, modality)
    
    file_list = os.listdir(hq_p) 
    for file_name in tqdm(file_list):
        hq_img = io.load(os.path.join(hq_p, file_name))['data']
        gen_img = io.load(os.path.join(gen_p, file_name))['data'] 
        eval_result = compute_measure([
            {
            "hq_img": hq_img.astype(np.float32), 
            "save_data": gen_img.astype(np.float32), 
            "modality": modality
                }
            ]) 
        
        result_dict["NAME"].append(file_name) 
        result_dict["PSNR"] += eval_result[0] 
        result_dict["SSIM"] += eval_result[1] 
        result_dict["RMSE"] += eval_result[2] 
    
    result_dict["PSNR_MEAN"] = np.mean(result_dict["PSNR"]) 
    result_dict["SSIM_MEAN"] = np.mean(result_dict["SSIM"]) 
    result_dict["RMSE_MEAN"] = np.mean(result_dict["RMSE"]) 
    io.save(
        {
            "save_data": result_dict, 
            "save_path": os.path.join(save_root, "{}.csv".format(modality))
            }
        ) 
    
    print("Modality: {}, Number: {}, PSNR: {:.6f}, SSIM: {:.6f}, RMSE: {:.6f}".format(modality, len(file_list), result_dict["PSNR_MEAN"], result_dict["SSIM_MEAN"], result_dict["RMSE_MEAN"] ))
        
        