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

# Step 13 - ddpm_train_step
import torch
import torch.nn.functional as F

def ddpm_train_step(params: dict, x0, schedule: dict, lr: float = 1e-2, seed: int = 0) -> tuple[dict, float]:
    # TODO: sample t,noise -> loss -> SGD on params
    
    B=x0.shape[0]
    t=schedule['T']
    alphas_cumprod=schedule['alphas_cumprod']

    torch.manual_seed(seed)

    t =torch.randint(0,t,(B,))
    noise=torch.randn_like(x0)

    loss=diffusion_training_loss(lambda x,t_:tiny_unet_forward(x,t_,params),x0,t,noise,alphas_cumprod)

    loss.backward()
    new_params={}

    with torch.no_grad():
        for k,v in params.items():
            if v.grad is not None:
                v_new=(v-lr*v.grad).detach().requires_grad_(True)
            else:
                v_new=v.detach().clone().requires_grad_(True)

            new_params[k]=v_new
        
    return new_params,float(loss.detach())


    
    pass

# Step 14 - train_ddpm
import torch
import torch.nn.functional as F

def train_ddpm(dataset, params: dict, schedule: dict, num_steps: int = 50, batch_size: int = 16, lr: float = 1e-2, seed: int = 0) -> tuple[dict, list]:
    # TODO: minibatch SGD training loop
    
    history=[]
    for i in range(num_steps):
        torch.manual_seed(seed+i)
        indice=torch.randint(0,len(dataset),(batch_size,))
        batch=dataset[indice]
        params,l=ddpm_train_step(params,batch,schedule,lr,seed)
    
        history.append(l)
    return params,history
    
    pass

# Step 15 - predict_x0_from_eps
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    # TODO: invert the q_sample equation for x0
    bar_alpha=extract_into_batch(alphas_cumprod,t,x_t)
    return (x_t-torch.sqrt(1-bar_alpha)*eps)/torch.sqrt(bar_alpha)
    pass

# Step 16 - ddpm_p_mean_variance
import torch
import torch.nn.functional as F

import torch

def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    alphas         = schedule['alphas']
    alphas_cumprod = schedule['alphas_cumprod']
    betas          = schedule['betas']

    # 预先补 1，处理 t=0
    alphas_cumprod_prev = torch.cat([
        torch.ones(1, device=alphas_cumprod.device, dtype=alphas_cumprod.dtype),
        alphas_cumprod[:-1]
    ], dim=0)

    alpha_bar_t    = extract_into_batch(alphas_cumprod,      t, x_t)
    alpha_bar_prev = extract_into_batch(alphas_cumprod_prev, t, x_t)
    beta_t         = extract_into_batch(betas,               t, x_t)
    alpha_t        = extract_into_batch(alphas,              t, x_t)

    x0_hat = predict_x0_from_eps(x_t, t, eps, alphas_cumprod)
    x0_hat = x0_hat.clamp(-1.0, 1.0)

    one_minus = 1.0 - alpha_bar_t
    coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / one_minus
    coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / one_minus

    mean = coef_x0 * x0_hat + coef_xt * x_t
    return mean, beta_t, x0_hat

# Step 17 - ddpm_p_sample
import torch
import torch.nn.functional as F

def ddpm_p_sample(x_t, t, params: dict, schedule: dict, noise=None):
    # TODO: one reverse step x_t -> x_{t-1}
    eps=tiny_unet_forward(x_t,t,params)
    mean,var,_=ddpm_p_mean_variance(x_t,t,eps,schedule)
    mask = (t.view(-1, 1, 1, 1) == 0)  
    if noise is None:
        noise = torch.randn_like(x_t)
    noise = torch.where(mask, torch.zeros_like(noise), noise)
    return mean+torch.sqrt(var)*noise
    
    pass

# Step 18 - ddpm_sample_loop
import torch
import torch.nn.functional as F

def ddpm_sample_loop(params: dict, schedule: dict, shape: tuple, seed: int = 0):
    # TODO: ancestral sampling from pure noise to x0
    torch.manual_seed(seed)
    T=schedule['T']
    B=shape[0]
    x = torch.randn(shape)
    for t in reversed(range(T)):
        t_batch=torch.full((B,),t)
        x=ddpm_p_sample(x,t_batch,params,schedule)
    return x
    pass

# Step 19 - sample_quality_mse
import torch
import torch.nn.functional as F

def sample_quality_mse(samples, dataset) -> float:
    # TODO: mean over samples of min MSE to any dataset image
    samples_flat = samples.reshape(samples.shape[0], -1)   # (N, D)
    dataset_flat = dataset.reshape(dataset.shape[0], -1)   # (M, D)

    min_mses = []
    for i in range(samples_flat.shape[0]):
        diff = dataset_flat - samples_flat[i]              # (M, D)
        mse = (diff ** 2).mean(dim=1)                      # (M,)
        min_mses.append(mse.min())
    return float(torch.stack(min_mses).mean())
    pass

# Step 20 - ddpm_experiment
import torch
import torch.nn.functional as F

def ddpm_experiment(n_data=64, size=8, T=20, hidden=16,
                    num_steps=40, batch_size=16, lr=5e-2,
                    n_samples=8, seed=0):
    dataset = make_blob_dataset(n_data, size, seed=seed)

    # 关键改动：显式用强 beta，让 T 较小时信号也能被破坏
    schedule = build_diffusion_schedule(T, beta_start=0.1, beta_end=0.4)

    params = init_tiny_unet(1, hidden, time_dim=hidden, seed=seed)

    params, history = train_ddpm(
        dataset, params, schedule, num_steps, batch_size, lr, seed
    )

    samples = ddpm_sample_loop(
        params, schedule, (n_samples, 1, size, size), seed=seed + 1
    )

    torch.manual_seed(seed + 2)
    noise = torch.randn(n_samples, 1, size, size)

    sample_mse = sample_quality_mse(samples, dataset)
    noise_mse = sample_quality_mse(noise, dataset)

    return {
        'train_losses': history,
        'final_loss': float(history[-1]),
        'sample_mse': float(sample_mse),
        'noise_mse': float(noise_mse),
        'improvement': float(noise_mse - sample_mse),
    }

