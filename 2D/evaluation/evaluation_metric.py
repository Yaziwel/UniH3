import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable

def compute_measure(data): 

    data_range_dict = { 
        "CT": [-1024.0, 3071.0], 
        "PET": [0.0, 20.0], 
        "MRI": [0.0, 4095.0], 
        "OCT": [0.0, 255.0],
        "Pathology": [0.0, 255.0],
        "Ultrasound": [0.0, 255.0],
        "X-ray": [0.0, 255.0],
        } 
    
    pred_psnr = [] 
    pred_ssim = [] 
    pred_rmse = []
    
    for i in range(len(data)):
        x = data[i]['save_data'] 
        pred = data[i]['hq_img'] 
        d_min, d_max = data_range_dict[data[i]['modality']]
        data_range = d_max - d_min
        pred_psnr.append(compute_PSNR(x, pred, data_range))
        pred_ssim.append(compute_SSIM(x, pred, data_range))
        pred_rmse.append(compute_RMSE(x, pred))
    return pred_psnr, pred_ssim, pred_rmse


def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        mse_ = compute_MSE(img1, img2)
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2)
        return 10 * np.log10((data_range ** 2) / mse_)


def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    if isinstance(img1, np.ndarray): 
        img1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor) 
        img2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)

        
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window