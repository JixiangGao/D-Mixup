import torch
import torch.nn as nn

def auto_cuda(x, use_cuda=True, gpu_device=0):
    if use_cuda and torch.cuda.is_available():
        if len(gpu_device) == 1:
            return x.cuda(int(gpu_device))
        else:
            device = torch.device("cuda:"+gpu_device[0])
            return x.to(device)
    else:
        return x
    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        try:
            nn.init.constant_(m.bias, 0.1)
        except:
            pass
    return
