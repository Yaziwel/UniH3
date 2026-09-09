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




def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def compute_SSIM(img1, img2, data_range, window_size = 11, size_average = True, use_cuda=True): 
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0) 
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0) 
    if use_cuda:
        img1 = img1.cuda() 
        img2 = img2.cuda()
    img1 = img1/data_range 
    img2 = img2/data_range 
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average).item()


