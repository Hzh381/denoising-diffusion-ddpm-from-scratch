"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    return torch.linspace(beta_start,beta_end,steps=T,dtype=torch.float32)
    pass

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    return 1-betas
    pass

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas,dim=0)
    pass

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    return  a[t].reshape(-1,1,1,1)
    pass

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    bar_alpha_t=extract_into_batch(alphas_cumprod,t,x0)
    return torch.sqrt(bar_alpha_t)*x0+torch.sqrt(1-bar_alpha_t)*noise
    pass

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    betas=linear_beta_schedule(T,beta_start,beta_end)
    alphas=alphas_from_betas(betas)
    alphas_cumprod=cumprod_alphas(alphas)
    sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod=torch.sqrt(1-alphas_cumprod)
    
    return {
        'T':T,
        'betas':betas,
        'alphas':alphas,
        'alphas_cumprod':alphas_cumprod,
        'sqrt_alphas_cumprod':sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod':sqrt_one_minus_alphas_cumprod
    }
    pass

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    return torch.mean((noise_pred-noise)**2)
    pass

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    x_t=q_sample(x0,t,noise,alphas_cumprod)
    noise_pred=model(x_t,t)
    return noise_prediction_loss(noise,noise_pred)
    pass

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    # TODO: sinusoidal timestep embedding of shape (B, dim)
    
    half=dim//2
    if half==1:
        exponent=torch.zeros(1)
    else:
        exponent=torch.arange(half)//(half-1)
    

    freg=10000**exponent
    arg=t.unsqueeze(1)/freg.unsqueeze(0)
    return torch.cat([torch.sin(arg),torch.cos(arg)],dim=1)

    pass

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    # TODO: initialize tiny residual denoiser parameters
    torch.manual_seed(seed)
    def make_weight(*shape):
        w=torch.randn(*shape)*0.02
        w.requires_grad_(True)
        return w
    def make_bias(*shape):
        b=torch.zeros(*shape)
        b.requires_grad_(True)
        return b

    return {
        'conv_in_w':make_weight(hidden,in_ch,3,3),
        'conv_in_b':make_bias(hidden),
        'time_mlp_w':make_weight(hidden,time_dim),
        'time_mlp_b':make_bias(hidden),
        'conv_mid_w':make_weight(hidden,hidden,3,3),
        'conv_mid_b':make_bias(hidden),
        'conv_out_w':make_weight(in_ch,hidden,3,3),
        'conv_out_b':make_bias(in_ch)
    }    
    
    
    pass

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # TODO: time-conditioned tiny CNN predicting noise
    h=F.conv2d(x,params['conv_in_w'],params['conv_in_b'],padding=1)
    temb=timestep_embedding(t,params['time_mlp_w'].shape[1])
    temb=F.linear(temb,params['time_mlp_w'],params['time_mlp_b'])
    temb=F.relu(temb)
    h=h+temb[:,:,None,None]
    h=F.relu(h)
    h=F.conv2d(h,params['conv_mid_w'],params['conv_mid_b'],padding=1)
    h=F.relu(h)
    return F.conv2d(h,params['conv_out_w'],params['conv_out_b'],padding=1)

    
    pass

# Step 12 - make_blob_dataset
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    # TODO: n images with a random bright disk on a black background
    torch.manual_seed(seed)
    radius=size//4
    images=torch.zeros(n,1,size,size)
    x=torch.arange(size).view(1,size)
    y=torch.arange(size).view(size,1)
    for i in range(n):
        center=torch.randint(radius,size-radius,(2,))
        cx,cy=center[0].item(),center[1].item()
        dist=(x-cx)**2+(y-cy)**2
        mask=dist<=radius**2
        images[i,0][mask]=1

    return images
        

    pass

# Step 13 - ddpm_train_step (not yet solved)
# TODO: implement

# Step 14 - train_ddpm (not yet solved)
# TODO: implement

# Step 15 - predict_x0_from_eps (not yet solved)
# TODO: implement

# Step 16 - ddpm_p_mean_variance (not yet solved)
# TODO: implement

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

