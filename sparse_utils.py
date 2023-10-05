import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Feather:
    def __init__(self, gth, theta):
        self.gth = gth
        self.theta = theta
    
    def forward(self, w):
        return Feather_aux.apply(w, self.gth, self.theta)
        


class Feather_aux(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, gth, theta):
             
        ctx.aux = torch.where(torch.abs(w) > gth, 1.0, theta)
     
        p = 3
        diff = torch.abs(w)**p - gth**p 
        w_masked = torch.where(diff > 0, torch.sign(w)*(diff)**(1/p), 0.0)   
        
        return w_masked

    @staticmethod
    def backward(ctx, g):
        
        g = ctx.aux*g           
        return g, None, None 
    

class SparseConv(nn.Module):
    def __init__(self, conv, feather):
        super(SparseConv, self).__init__()
        self.conv = conv
        self.feather = feather

    def forward(self, x):

        w = self.conv.weight
        b = self.conv.bias
        stride = self.conv.stride
        padding = self.conv.padding
        groups = self.conv.groups


        if self.feather.gth > 0:
            w = self.feather.forward(w)

        out = F.conv2d(x, w, bias=b, padding=padding, stride=stride, groups=groups)
        return out

class SparseFc(nn.Module):
    def __init__(self, fc, feather):
        super(SparseFc, self).__init__()
        self.fc = fc
        self.feather = feather
 
    def forward(self, x):

        w = self.fc.weight
        b = self.fc.bias
                
        if self.feather.gth > 0:            
            w = self.feather.forward(w)
            
        out = F.linear(x, w, bias=b)
        return out


def iter_sparsify(m, feather, pthres=0):
    for name, child in m.named_children():
        iter_sparsify(child, feather, pthres)
        if type(child) == nn.Conv2d:
            nw = (child.in_channels * child.out_channels * child.kernel_size[0] * child.kernel_size[1]) / child.groups
            if nw >= pthres:  
                slayer = SparseConv(child, feather)
                m.__setattr__(name, slayer)

              
        if type(child) == nn.Linear:
            nw = child.in_features * child.out_features
            if nw >= pthres:
                slayer = SparseFc(child, feather)
                m.__setattr__(name, slayer)


def iter_desparsify(m, gth):
    for name, child in m.named_children():
        iter_desparsify(child, gth)
        if type(child) == SparseConv:
            conv = child.conv
            w = conv.weight.data

            nw = F.hardshrink(w, gth)
            conv.weight.data = nw

            m.__setattr__(name, conv)

        if type(child) == SparseFc:
            fc= child.fc
            w = fc.weight.data

            nw = F.hardshrink(w, gth)
            fc.weight.data = nw

            m.__setattr__(name, fc)

 
def get_params(model):
    bn_ids =[]  
    modules = list(model.named_modules())      
    for n, layer in modules:
        if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d): 
            bn_ids.append(id(layer.weight))
            bn_ids.append(id(layer.bias))
    
    params, params_nowd = [], []
    for name, p in model.named_parameters():
    
        if id(p) in bn_ids or 'bias' in name:
            params_nowd += [p]
        else:
            params += [p]
    return params, params_nowd


def get_prunable_weights_cnt(model):
    prunable_weights_cnt = 0
    temp_dims = [0]
    for name, layer in model.named_modules():
        if ('Sparse' in layer.__class__.__name__):
            if 'Conv' in layer.__class__.__name__ :
                w = layer.conv.weight
            elif 'Fc' in layer.__class__.__name__:
                w = layer.fc.weight
            else:
                print(" Not Recognized Sparse Module ")

            temp_dims.append(w.numel())         
            tnum = w.numel()
            prunable_weights_cnt += tnum
            
    idx_list = [0]
    for i in range(len(temp_dims)):
        idx_list.append(temp_dims[i] + idx_list[i])
            
    return prunable_weights_cnt, idx_list


def calc_thresh(w, ratio):
    w_sorted, _ = torch.sort(w)
    m = (len(w_sorted)-1)*ratio
    idx_floor, idx_ceil = int(np.floor(m)), int(np.ceil(m))
    v1, v2 = w_sorted[idx_floor], w_sorted[idx_ceil]
    thresh = v1 + (v2-v1)*(m-idx_floor)
    return thresh.item()



def get_global_thresh(model, device, st_batch, prunable_weights_cnt, idx_list):
    i = 1
    w_total = torch.empty(prunable_weights_cnt).to(device) 
    for name, layer in model.named_modules():
        if ('Sparse' in layer.__class__.__name__):
            if 'Conv' in layer.__class__.__name__ :
                w = layer.conv.weight.flatten().detach()            
            elif 'Fc' in layer.__class__.__name__:
                w = layer.fc.weight.flatten().detach() 

            w_total[idx_list[i] : idx_list[i+1]] = w
            i +=1
         
    global_thresh = calc_thresh(torch.abs(w_total), st_batch)
    
    return global_thresh


def pruning_scheduler(final_rate, nbatches, ntotalsteps, t1):
    
    kf = final_rate
    t1 = t1*nbatches   
    t2 = int(np.floor(ntotalsteps*0.5))

    t = np.arange(t1,t2)
    k = np.hstack(( np.zeros(t1), ( kf - kf*(1-(t-t1)/(t2-t1))**3), kf*np.ones(ntotalsteps-t2) )) 

    return k
 
def get_theta(final_rate):
    if final_rate > 0.95:
        theta = 0.5
    else:
        theta = 1.0
    return theta


class Pruner:  
    def __init__(self, model, device, final_rate,  nbatches, epochs, pthres=0, t1=0):

        theta = get_theta(final_rate)
        self.ntotalsteps = nbatches*epochs
        self.step_idx = 0
        
        self.feather = Feather(gth=0.0, theta=theta)     
        self.device = device
        self.t1 = t1
        
        self.model = model 
           
        iter_sparsify(m=self.model, feather=self.feather, pthres=pthres)     
        prunable_weights_cnt, idx_list = get_prunable_weights_cnt(self.model)
        
        self.prunable_weights_cnt = prunable_weights_cnt
        self.idx_list = idx_list
        
        pscheduler = pruning_scheduler(final_rate, nbatches, self.ntotalsteps, self.t1)
        self.pscheduler = pscheduler
        
    def update_thresh(self, end_of_batch=False):       
        idx = self.step_idx
        if end_of_batch:
            idx -=1
        st_batch = self.pscheduler[idx]
        
        new_gth = 0.0
        if st_batch > 0:
            new_gth = get_global_thresh(self.model, self.device, st_batch, self.prunable_weights_cnt, self.idx_list)

        self.feather.gth = new_gth 
        if not end_of_batch:
            self.step_idx += 1 
        
    def print_sparsity(self):   
        local_zeros_cnt = 0
        for name, layer in self.model.named_modules():
            if ('Sparse' in layer.__class__.__name__):
                if 'Conv' in layer.__class__.__name__ :
                    w = layer.conv.weight
                elif 'Fc' in layer.__class__.__name__:
                    w = layer.fc.weight
                else:
                    print(" Not Recognized Sparse Module ")

                th = self.feather.gth 
                nw = F.hardshrink(w, th)
                      
                tsparsity = (nw == 0).float().sum().item()
                local_zeros_cnt += tsparsity
        
                tnum = nw.numel()
    
                print(f'{name}'.ljust(40), f'#w: {int(tnum)}'.ljust(11), f'| sparsity: {round(100.0 * tsparsity / tnum, 2)}%'.ljust(18))  
                  
        return 100 * float(local_zeros_cnt) / float(self.prunable_weights_cnt)
    
    def desparsify(self):
        gth = self.feather.gth 
        iter_desparsify(self.model, gth)












































