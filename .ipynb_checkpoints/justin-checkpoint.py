#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F  
import matplotlib 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.utils.data
from torch.utils.data import DataLoader
import random
import math

EPSILON = 0.1       # Max perturbation (for L∞ PGD)
ALPHA = 0.01         # Step size per iteration
ATTACK_ITERATIONS = 40
TARGET_LABEL = 2     # Example target label for the targeted attack

# System/Model parameters
sys.path.append("/home/jfeng/Desktop/jfeng/rf_spoofing/spoofing/models")
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/jfeng/Desktop/jfeng/rf_spoofing/spoofing/weights/best_model_retrained.pth"
#IQ_FILE_PATH = "/home/jfeng/Desktop/jfeng/rf_spoofing/spoofing/1m_2m_replacedPluto4/Pluto_10_windows_runs2_3/Pluto_10_2m_run3.iq"
# This is the file i was usingn before May 27
#IQ_FILE_PATH = "/data2/brandan/RFSPOOF/May22Test/Pluto10_2m.iq"
IQ_FILE_PATH = "/home/jfeng/Desktop/May27/Pluto_10_test.iq"

from attempt2 import resnet50_1d  # Directly import from attempt2.py
num_classes = 8  # Change this if your model was trained with a different number of classes

# Initialize the model architecture
model = resnet50_1d(num_classes=num_classes).to(DEVICE)

# Load trained weights
print(f"Loading trained model weights from: {MODEL_PATH}")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()


# In[2]:


class IQDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        sample = torch.from_numpy(sample).float()
        # Normalize data
        magnitude = torch.sqrt(torch.sum(sample**2, dim=1, keepdim=True))
        sample = sample / magnitude

        label_tensors = torch.tensor(label, dtype=torch.long)

        return sample, label_tensors


# In[3]:


BATCH_SIZE = 16
WINDOW_SIZE = 10000
HOP_SIZE = 500
START_INDEX = 4800
END_INDEX = 6000

def path_loss(signal, n, d1, d2):
    signal = signal
    scaling = (d1 / d2) ** (n / 2)
    return signal * scaling

def apply_rician_fading(signal, K=10.0):
    """
    Apply Rician fading to the signal, which is of shape [2, N] (I and Q).
    Fading is applied using a K-factor (default 10).
    """
    i_window = signal[:, 0, :]
    q_window = signal[:, 1, :]

    device = signal.device
    dtype = signal.dtype
    # Calculate scaling factors
    K = torch.tensor(K, dtype=dtype, device=device)
    scale_LOS = torch.sqrt(K / (K + 1))
    scale_NLOS = torch.sqrt(1 / (K + 1))

    # Generate NLOS (Rayleigh) component
    real_nlos = torch.randn(i_window.shape, device=device) / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    imag_nlos = torch.randn(i_window.shape, device=device) / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))

    # LOS component (typically assumed as 1 + 0j for all samples)
    real_los = torch.ones(i_window.shape, device=device)
    imag_los = torch.zeros(i_window.shape, device=device)

    # Total fading coefficients
    real_fade = scale_LOS * real_los + scale_NLOS * real_nlos
    imag_fade = scale_LOS * imag_los + scale_NLOS * imag_nlos

    # Apply Rician fading
    faded_real = i_window * real_fade - q_window * imag_fade
    faded_imag = i_window * imag_fade + q_window * real_fade

    # Reconstruct the faded signal back into a tensor
    faded_signal = torch.stack((faded_real, faded_imag), dim=1)

    return faded_signal

def apply_awgn(signal, noise_std=0.000001):
    device = signal.device
    dtype = signal.dtype
    # Split noise equally between I and Q (to maintain total variance)
    per_dim_std = noise_std / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    # Generate i.i.d. Gaussian noise for I and Q
    noise = torch.randn_like(signal, device=device) * per_dim_std
    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal

def transform_channel_effects(x, chosen_distance=2.0, path_loss_exponent=2.0, reference_distance=1.0, noise_std=0.000001, k=10.0):   
    #print("Signal original: ", x) 
    signal_path_loss = path_loss(x, path_loss_exponent, reference_distance, chosen_distance)
    #print("Signal Path Loss: ", signal_path_loss)
    signal_rician = apply_rician_fading(signal_path_loss, k)
    #print("Signal Rician: ", signal_rician)
    signal_awgn = apply_awgn(signal_rician, noise_std)
    #print("Signal AWGN: ", signal_awgn)
    return signal_awgn

# In[4]:


DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def apply_iq_imbalances_torch(x, G_I=1.0, G_Q=1.0, phi=0.0, dc_I=0.0, dc_Q=0.0):
    """
    Apply gain imbalance, phase imbalance, and DC offset to IQ data using PyTorch.

    Parameters:
    - x : torch.Tensor, shape (B, 2, N)
        Input IQ signal, where x[:, 0, :] = I and x[:, 1, :] = Q.
    - G_I : float or torch.Tensor
        Gain applied to I channel.
    - G_Q : float or torch.Tensor
        Gain applied to Q channel.
    - phi : float
        Phase imbalance in radians.
    - dc_I : float or torch.Tensor
        DC offset added to I channel.
    - dc_Q : float or torch.Tensor
        DC offset added to Q channel.

    Returns:
    - x_out : torch.Tensor, shape (B, 2, N)
        IQ signal after applying impairments.
    """
    I = x[:, 0, :]
    Q = x[:, 1, :]

    # DC Offset
    I_offset = I + dc_I
    Q_offset = Q + dc_Q

    # Gain Imbalance
    gain_ratio = G_Q / G_I
    I_gain = I_offset
    Q_gain = Q_offset * gain_ratio

    # Phase
    phi_tensor = torch.tensor(phi)
    I_final = I_gain
    Q_final = Q_gain * torch.cos(phi_tensor) + I_gain * torch.sin(phi_tensor)

    x_out = torch.stack([I_final, Q_final], dim=1)
    
    return x_out

def targeted_eot_pgd_attack(
    model,
    x,
    perturb,
    y,
    dc_offset_I_list,
    dc_offset_Q_list,
    gain_I_list, 
    gain_Q_list,
    phase_imbalance_list,
    target_label,
    eps=0.1,
    alpha=0.01,
    num_iter=40,
    num_samples=10,
    min_distance=1.0,
    max_distance=4.0,
    path_loss_exponent=2.0,
    reference_distance=1.0,
    min_noise_std=0.000001,
    max_noise_std=0.0001,
    min_k=10,
    max_k=20,
):
    x_adv = perturb.clone().detach().requires_grad_(True).to(DEVICE)
    target = torch.full_like(y, target_label)

    best_x_adv = x_adv.clone().detach()
    best_target_confidence = -float('inf')
    # PGD Iterations
    for i in range(num_iter):
        total_grad = torch.zeros_like(x_adv)
        grads = []
        # EOT Iterations
        for _ in range(num_samples):
            all_logits = []
            # Choose a random distance between min and max, for this EOT iteration
            chosen_distance = torch.empty(1).uniform_(min_distance, max_distance + 0.0000001).item()
            chosen_k = torch.empty(1).uniform_(min_k, max_k + 0.0000001).item()
            chosen_noise_std = torch.empty(1).uniform_(min_noise_std, max_noise_std + 0.0000001).item()

            # DC Offset and IQ Imbalance
            
            random_index = random.randint(0,9)
            dc_offset_I = dc_offset_I_list[random_index]
            dc_offset_Q = dc_offset_Q_list[random_index]
            gain_I = gain_I_list[random_index]
            gain_Q = gain_Q_list[random_index]
            phase_imbalance = phase_imbalance_list[random_index]
            
            #dc_offset_I = -0.000005
            #dc_offset_Q = -0.000005
            #gain_I = 0.04
            #gain_Q = 0.04002
            #phase_imbalance = -0.00004

            #x_scaled = x_adv
            # Apply real-world transmitter impairments
            #x_adjust = x_adv
            x_adjust = apply_iq_imbalances_torch(x_adv, gain_I, gain_Q, phase_imbalance, dc_offset_I, dc_offset_Q)
            
            x_t = transform_channel_effects(
                x_adjust,
                chosen_distance=chosen_distance,
                path_loss_exponent=path_loss_exponent,
                reference_distance=reference_distance,
                noise_std=chosen_noise_std,
                k=chosen_k
            )
            
            # Grab random x
            idx = random.randint(0, x.shape[-1] - 10000)
            x_tx = x[:, :, idx:idx+10000].to(DEVICE)
            #x_tx = x[:, :, 0:10000].to(DEVICE)

            x_combined = x_tx.detach() + x_t

            logits = model(x_combined)
            all_logits.append(logits)
            loss = F.cross_entropy(logits, target)

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            grads.append(grad.detach())
            total_grad += grad

        grads_tensor = torch.stack(grads)
        grad_variance = grads_tensor.var(dim=0).mean().item()
        print(f"PGD step {i + 1}/{num_iter}: gradient variance = {grad_variance:.4e}")
        avg_grad = total_grad / num_samples

        with torch.no_grad():
            x_adv -= alpha * avg_grad.sign()
            #x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.clamp(x_adv, -eps, eps)
            x_adv = x_adv.detach().clone().requires_grad_(True)

        with torch.no_grad():
            avg_logits = torch.stack(all_logits).mean(dim=0)
            target_confidence = F.softmax(avg_logits, dim=1)[:, target_label]  # shape: [batch]
            avg_conf = target_confidence.mean().item()

            if avg_conf > best_target_confidence:
                best_target_confidence = avg_conf
                best_x_adv = x_adv.clone().detach()

        with torch.no_grad():
            # Choose a random distance between min and max, for this EOT iteration
            chosen_distance = torch.empty(1).uniform_(min_distance, max_distance + 0.0000001).item()
            chosen_k = torch.empty(1).uniform_(min_k, max_k + 0.0000001).item()
            chosen_noise_std = torch.empty(1).uniform_(min_noise_std, max_noise_std + 0.0000001).item()
            
            # DC offset and IQ imbalance
            
            random_index = random.randint(0,9)
            dc_offset_I = dc_offset_I_list[random_index]
            dc_offset_Q = dc_offset_Q_list[random_index]
            gain_I = gain_I_list[random_index]
            gain_Q = gain_Q_list[random_index]
            phase_imbalance = phase_imbalance_list[random_index]
            

            #dc_offset_I = -0.000005
            #dc_offset_q = -0.000005
            #gain_I = 0.04
            #gain_Q = 0.04002
            #phase_imbalance = -0.0004
            
            #x_scaled = x_adv
            # Apply real-world transmitter impairments
            x_adjust = apply_iq_imbalances_torch(x_adv, gain_I, gain_Q, phase_imbalance, dc_offset_I, dc_offset_Q)
            
            x_t = transform_channel_effects(
                x_adjust,
                chosen_distance=chosen_distance,
                path_loss_exponent=path_loss_exponent,
                reference_distance=reference_distance,
                noise_std=chosen_noise_std,
                k=chosen_k
            )
            
            # Grab random x
            idx = random.randint(0, x.shape[-1] - 10000)
            x_tx = x[:, :, idx:idx+10000].to(DEVICE)
            #x_tx = x[:, :, 0:10000].to(DEVICE)
            
            x_combined = x_t + x_tx

            pred = model(x_combined).argmax(dim=1)
            print(pred)
            success = (pred == target).float().mean().item()
            print(f"Attack success rate: {success * 100:.2f}%")

    return best_x_adv


# In[10]:


def generate_eot_pgd_noise():
    print(f"Loading data from: {IQ_FILE_PATH}")
    label = 10
    print(f"True label: {label}, Target label: {TARGET_LABEL}")

    # Load IQ data
    data = np.fromfile(IQ_FILE_PATH, dtype="float32")
    real = data[0::2]
    imag = data[1::2]
    #start = (START_INDEX + 1) * HOP_SIZE
    #end = start + WINDOW_SIZE
    #i_window = real[start:end]
    #q_window = imag[start:end]
    combined = np.vstack((real, imag))  # [2, N]

    # Format into dataset
    test_dataset = IQDataset([combined], [label])
    data_tensor, label_tensor = test_dataset[0]
    data_tensor = data_tensor.unsqueeze(0).to(DEVICE)
    
    #label_tensor = label_tensor.unsqueeze(0).to(DEVICE)
    #----------------------------------------------------------
    perturbation_tensor = torch.empty((1, 2, WINDOW_SIZE), dtype=torch.float32).uniform_(-0.1, 0.1).to(DEVICE)
    label_tensor = torch.tensor([label], dtype=torch.long).to(DEVICE)
    #----------------------------------------------------------
    min_distance=2.0
    max_distance=8.0
    min_k = 10
    max_k = 20
    min_noise_std = 0.000001
    max_noise_std = 0.001
    path_loss_exponent=2.0
    reference_distance=1.0
    EPSILON = 0.1
    ALPHA = 0.01
    ATTACK_ITERATIONS = 60
    if min_distance >= 5.0:
        EPSILON = 0.6
        ALPHA = 0.021
        ATTACK_ITERATIONS = 100
    
    
    dc_offset_I_list = [-0.0000048145, -0.0000050625, -0.0000049675, -0.0000049891, -0.0000050236, -0.0000049773, -0.0000052696, -0.0000050665, -0.0000049802, -0.0000050692]
    dc_offset_Q_list = [-0.0000048619, -0.0000049788, -0.0000049042, -0.0000048843, -0.0000048275, -0.0000051674, -0.0000050298, -0.0000046535, -0.0000049399, -0.0000049609]
    gain_I_list = [0.04946, 0.02505, 0.00776, 0.00737, 0.03989, 0.04072, 0.04677, 0.06374, 0.01650, 0.01540]
    gain_Q_list = [0.04948, 0.02504, 0.00775, 0.00737, 0.03991, 0.04074, 0.04675, 0.06376, 0.01649, 0.01540]
    phase_imbalance_list = [-0.00058, 0.00060, -0.00013, 0.00025, -0.00045, -0.00078, 0.00066, -0.00046, 0.00029, -0.00044]
    
    '''
    dc_offset_I_list = [-0.0000048145, -0.0000050625, -0.0000049675, -0.0000049891, -0.0000050236, -0.0000049773, -0.0000052696, -0.0000050665, -0.0000049802, -0.0000050692, -0.0000050020, -0.0000051169, -0.0000049720, -0.0000050209, -0.0000051375, -0.0000050125, -0.0000050764, -0.0000048907, -0.0000049981]
    dc_offset_Q_list = [-0.0000048619, -0.0000049788, -0.0000049042, -0.0000048843, -0.0000048275, -0.0000051674, -0.0000050298, -0.0000046535, -0.0000049399, -0.0000049609, -0.0000049763, -0.0000049312, -0.0000048910, -0.0000049030, -0.0000049336, -0.0000049982, -0.0000049354, -0.0000048764, -0.0000049646]
    gain_I_list = [0.04946, 0.02505, 0.00776, 0.00737, 0.03989, 0.04072, 0.04677, 0.06374, 0.01650, 0.01540, 0.02228, 0.01509, 0.01450, 0.01307, 0.02087, 0.00689, 0.01727, 0.01944, 0.00888]
    gain_Q_list = [0.04948, 0.02504, 0.00775, 0.00737, 0.03991, 0.04074, 0.04675, 0.06376, 0.01649, 0.01540, 0.02229, 0.01511, 0.01450, 0.01306, 0.02088, 0.00690, 0.01727, 0.01943, 0.00888]
    phase_imbalance_list = [-0.00058, 0.00060, -0.00013, 0.00025, -0.00045, -0.00078, 0.00066, -0.00046, 0.00029, -0.00044, -0.00070, 0.00036, 0.00069, 0.00038, -0.00049, -0.00144, -0.00055, 0.00083, 0.00007]
'''
    
    # EoT PGD attack
    x_adv = targeted_eot_pgd_attack(
        model=model,
        x=data_tensor,
        perturb=perturbation_tensor,
        y=label_tensor,
        dc_offset_I_list=dc_offset_I_list,
        dc_offset_Q_list = dc_offset_Q_list,
        gain_I_list=gain_I_list,
        gain_Q_list=gain_Q_list,
        phase_imbalance_list=phase_imbalance_list,
        target_label=TARGET_LABEL,
        eps=EPSILON,
        alpha=ALPHA,
        num_iter=ATTACK_ITERATIONS,
        num_samples=10,
        min_distance=min_distance,
        max_distance=max_distance,
        path_loss_exponent=path_loss_exponent,
        reference_distance=reference_distance,
        min_noise_std=min_noise_std,
        max_noise_std=max_noise_std,
        min_k=min_k,
        max_k=max_k,
    )


    noise_np = x_adv.squeeze().detach().cpu().numpy()

    interleaved_noise = np.empty(noise_np.shape[1] * 2, dtype=np.float32)
    interleaved_noise[0::2] = noise_np[0]
    interleaved_noise[1::2] = noise_np[1]

    interleaved_noise = interleaved_noise
    
    magnitude = np.mean(np.abs(interleaved_noise[0::2] + interleaved_noise[1::2] * 1j))
    print("Magnitude: ", magnitude)

    # write out the noise file
    save_path = "May29/perturbation/acc.iq"
    interleaved_noise.tofile(save_path)
    print(f"Saved EoT PGD *noise* to {save_path}")


generate_eot_pgd_noise()


# In[10]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def targeted_eot_pgd_attack2(
    model,
    x,
    perturb,
    y,
    target_label,
    eps=0.1,
    alpha=0.01,
    num_iter=40,
    num_samples=10,
    min_distance=1.0,
    max_distance=4.0,
    path_loss_exponent=2.0,
    reference_distance=1.0,
    min_noise_std=0.000001,
    max_noise_std=0.0001,
    min_k=10,
    max_k=20
):
    x_acc = perturb.clone().detach().requires_grad_(True).to(DEVICE)
    target = torch.full_like(y, target_label)

    best_x_acc = x_acc.clone().detach()
    best_target_confidence = -float('inf')
    # PGD Iterations
    for i in range(num_iter):
        total_grad = torch.zeros_like(x_acc)
        grads = []
        # EOT Iterations
        for _ in range(num_samples):
            all_logits = []
            # Choose a random distance between min and max, for this EOT iteration
            chosen_distance = torch.empty(1).uniform_(min_distance, max_distance + 0.0000001).item()
            chosen_k = torch.empty(1).uniform_(min_k, max_k + 0.0000001).item()
            chosen_noise_std = torch.empty(1).uniform_(min_noise_std, max_noise_std + 0.0000001).item()
            x_t = transform_channel_effects(
                x_acc,
                chosen_distance=chosen_distance,
                path_loss_exponent=path_loss_exponent,
                reference_distance=reference_distance,
                noise_std=chosen_noise_std,
                k=chosen_k
            )
            d_tx   = float(torch.empty(1).uniform_(min_distance, max_distance))
            sigma_tx   = float(torch.empty(1).uniform_(min_noise_std, max_noise_std))
            k_tx   = float(torch.empty(1).uniform_(min_k, max_k))
            x_tx_t = transform_channel_effects(
                x,
                chosen_distance=d_tx,
                path_loss_exponent=path_loss_exponent,
                reference_distance=reference_distance,
                noise_std=sigma_tx,
                k=k_tx
            )
            x_combined = x_t + x_tx_t
            logits = model(x_combined)
            all_logits.append(logits)
            loss = F.cross_entropy(logits, target)

            grad = torch.autograd.grad(loss, x_acc, retain_graph=False, create_graph=False)[0]
            grads.append(grad.detach())
            total_grad += grad

        grads_tensor = torch.stack(grads)
        grad_variance = grads_tensor.var(dim=0).mean().item()
        print(f"PGD step {i + 1}/{num_iter}: gradient variance = {grad_variance:.4e}")
        avg_grad = total_grad / num_samples

        with torch.no_grad():
            x_acc -= alpha * avg_grad.sign()
            x_acc = torch.max(torch.min(x_acc, x + eps), x - eps)
            x_acc = x_acc.detach().clone().requires_grad_(True)

        with torch.no_grad():
            avg_logits = torch.stack(all_logits).mean(dim=0)
            target_confidence = F.softmax(avg_logits, dim=1)[:, target_label]  # shape: [batch]
            avg_conf = target_confidence.mean().item()

            if avg_conf > best_target_confidence:
                best_target_confidence = avg_conf
                best_x_acc = x_acc.clone().detach()

        with torch.no_grad():
            # Choose a random distance between min and max, for this EOT iteration
            chosen_distance = torch.empty(1).uniform_(min_distance, max_distance + 0.0000001).item()
            chosen_k = torch.empty(1).uniform_(min_k, max_k + 0.0000001).item()
            chosen_noise_std = torch.empty(1).uniform_(min_noise_std, max_noise_std + 0.0000001).item()
            x_t = transform_channel_effects(
                x_acc,
                chosen_distance=chosen_distance,
                path_loss_exponent=path_loss_exponent,
                reference_distance=reference_distance,
                noise_std=chosen_noise_std,
                k=chosen_k
            )
            x_combined = x_t + x
            pred = model(x_combined).argmax(dim=1)
            success = (pred == target).float().mean().item()
            print(f"Attack success rate: {success * 100:.2f}%")

    return best_x_acc


# In[ ]:




