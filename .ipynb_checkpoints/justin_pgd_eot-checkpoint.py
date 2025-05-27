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

EPSILON = 0.1       # Max perturbation (for L∞ PGD)
ALPHA = 0.01         # Step size per iteration
ATTACK_ITERATIONS = 40
TARGET_LABEL = 1     # Example target label for the targeted attack

# System/Model parameters
sys.path.append("/home/jfeng/Desktop/jfeng/rf_spoofing/spoofing/models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/jfeng/Desktop/jfeng/rf_spoofing/spoofing/weights/best_model_retrained.pth"
IQ_FILE_PATH = "/home/jfeng/Desktop/jfeng/rf_spoofing/spoofing/1m_2m_replacedPluto4/Pluto_10_windows_runs2_3/Pluto_10_2m_run3.iq"

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


# In[4]:


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


# In[46]:


TRUE_IQ_FILE_PATH = "/home/jfeng/Desktop/jfeng/rf_spoofing/spoofing/1m_2m_replacedPluto4/1m_2m/Pluto_1_2m_run3.iq"
BATCH_SIZE = 16
WINDOW_SIZE = 10000
HOP_SIZE = 500
START_INDEX = 4800
END_INDEX = 6000

# Initialize and load model
model = resnet50_1d(num_classes=8)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


fname = os.path.basename(TRUE_IQ_FILE_PATH)
print(f"\nProcessing file: {TRUE_IQ_FILE_PATH}")

# Extract label from filename: Pluto_#_...
try:
    target_label = int(fname.split("_")[1])
except (IndexError, ValueError):
    print(f"  Warning: could not parse label from filename: {fname}")
    

# Load IQ data
data = np.fromfile(TRUE_IQ_FILE_PATH, dtype="float32")
real_part = data[0::2]
imag_part = data[1::2]

test_data_tensors = []
test_label_tensors = []

# Sliding window generation
for x in range(START_INDEX, END_INDEX):
    start = (x + 1) * HOP_SIZE
    end = start + WINDOW_SIZE
    if end > len(real_part): break

    i_window = real_part[start:end]
    q_window = imag_part[start:end]
    combined = np.vstack((i_window, q_window))  # [2, WINDOW_SIZE]
    test_data_tensors.append(combined)
    test_label_tensors.append(target_label)

if not test_data_tensors:
    print(f"Skipping {fname}: not enough valid IQ segments.")
    

# Stack and shuffle
test_data_tensors = np.stack(test_data_tensors, axis=0)
test_label_tensors = np.array(test_label_tensors)
indices = np.random.permutation(len(test_data_tensors))
test_data = test_data_tensors[indices]
test_labels = test_label_tensors[indices]

# Create Dataset + Loader
test_dataset = IQDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluate
correct = 0
total = 0
mismatch_count = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        mismatch_count += (predicted != labels).sum().item()
        correct += (predicted == labels).sum().item()

acc = correct / total * 100
print(f"[{fname}] Accuracy: {acc:.2f}% | Mismatches: {mismatch_count} / {total}")

def targeted_pgd_attack(model, x, y, target_label, eps, alpha, num_iter):

    model.eval()
    
    if isinstance(target_label, int):
        target_label = torch.full_like(y, target_label)
    
    x_adv = x.clone().detach().to(DEVICE)
    x_adv.requires_grad = True

    for _ in range(num_iter):
        outputs = model(x_adv)
        
        loss = -nn.CrossEntropyLoss()(outputs, target_label)

        model.zero_grad()
        loss.backward()

        grad = x_adv.grad.data
        x_adv = x_adv.detach() + alpha * grad.sign()

        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)

        x_adv.requires_grad = True

    return x_adv.detach()


# In[48]:


def main():
    print(f"Loading data from: {IQ_FILE_PATH}")

    # Use true label from filename (Pluto_0 → class 0)
    label = int(os.path.basename(IQ_FILE_PATH).split("_")[1])  # This will be 0
    print(f"True label: {label}, Target label: {TARGET_LABEL}")

    # Load and format IQ data
    data = np.fromfile(IQ_FILE_PATH, dtype="float32")
    real = data[0::2]
    imag = data[1::2]

    start = (START_INDEX + 1) * HOP_SIZE
    end = start + WINDOW_SIZE
    i_window = real[start:end]
    q_window = imag[start:end]
    combined = np.vstack((i_window, q_window))  # [2, N]

    # Wrap in Dataset for normalization
    test_dataset = IQDataset([combined], [label])
    data_tensor, label_tensor = test_dataset[0]
    data_tensor = data_tensor.unsqueeze(0).to(DEVICE)
    label_tensor = label_tensor.unsqueeze(0).to(DEVICE)

    # Prediction before attack
    with torch.no_grad():
        logits = model(data_tensor)
        probs = F.softmax(logits, dim=1)
        orig_pred = torch.argmax(probs, dim=1)
        orig_conf = probs.max(dim=1).values

    print(f"Original prediction: {orig_pred.item()}, Confidence: {orig_conf.item():.4f}")

    # PGD attack targeting label 1
    x_adv = targeted_pgd_attack(
        model=model,
        x=data_tensor,
        y=label_tensor,
        target_label=TARGET_LABEL,
        eps=EPSILON,
        alpha=ALPHA,
        num_iter=ATTACK_ITERATIONS
    )

    with torch.no_grad():
        logits_adv = model(x_adv)
        probs_adv = F.softmax(logits_adv, dim=1)
        adv_pred = torch.argmax(probs_adv, dim=1)
        adv_conf = probs_adv.max(dim=1).values

    print(f"Adversarial prediction: {adv_pred.item()}, Confidence: {adv_conf.item():.4f}")

    # Save perturbation
    original_np = data_tensor.squeeze().cpu().numpy()
    adv_np = x_adv.squeeze().cpu().numpy()
    I_diff = adv_np[0] - original_np[0]
    Q_diff = adv_np[1] - original_np[1]

    interleaved = np.empty(I_diff.size + Q_diff.size, dtype=np.float32)
    interleaved[0::2] = I_diff
    interleaved[1::2] = Q_diff

    save_path = "May1/pluto10_to_target1_noise.iq"
    interleaved.tofile(save_path)
    print(f"Saved perturbation to {save_path}")

main()

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def path_loss(signal, n, d1, d2):
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

def transform_channel_effects(x, chosen_distance=2.0, path_loss_exponent=2.0, reference_distance=1.0, noise_std=0.000001, k=20.0):   
    #print("Signal original: ", x)
    signal_path_loss = path_loss(x, path_loss_exponent, reference_distance, chosen_distance)
    #print("Signal Path Loss: ", signal_path_loss)
    signal_rician = apply_rician_fading(signal_path_loss, k)
    #print("Signal Rician: ", signal_rician)
    signal_awgn = apply_awgn(signal_rician, noise_std)
    #print("Signal AWGN: ", signal_awgn)
    return signal_awgn


def targeted_eot_pgd_attack(
    model,
    x,
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
    noise_std=0.01,
    k=20.0
):
    x_adv = x.clone().detach().requires_grad_(True).to(DEVICE)
    target = torch.full_like(y, target_label)
    # PGD Iterations
    for i in range(num_iter):
        total_grad = torch.zeros_like(x_adv)
        # EOT Iterations
        for _ in range(num_samples):
            # Choose a random distance between min and max, for this EOT iteration
            chosen_distance = torch.empty(1).uniform_(min_distance, max_distance + 0.0000001).item()
            x_t = transform_channel_effects(
                x_adv,
                chosen_distance=chosen_distance,
                path_loss_exponent=path_loss_exponent,
                reference_distance=reference_distance,
                noise_std=noise_std,
                k=k
            )

            logits = model(x_t)
            loss = F.cross_entropy(logits, target)

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            total_grad += grad

        avg_grad = total_grad / num_samples

        with torch.no_grad():
            x_adv -= alpha * avg_grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = x_adv.detach().clone().requires_grad_(True)

    return x_adv

def generate_eot_pgd_noise():
    print(f"Loading data from: {IQ_FILE_PATH}")
    label = int(os.path.basename(IQ_FILE_PATH).split("_")[1])
    print(f"True label: {label}, Target label: {TARGET_LABEL}")

    # Load IQ data
    data = np.fromfile(IQ_FILE_PATH, dtype="float32")
    real = data[0::2]
    imag = data[1::2]
    start = (START_INDEX + 1) * HOP_SIZE
    end = start + WINDOW_SIZE
    i_window = real[start:end]
    q_window = imag[start:end]
    combined = np.vstack((i_window, q_window))  # [2, N]

    # Format into dataset
    test_dataset = IQDataset([combined], [label])
    data_tensor, label_tensor = test_dataset[0]
    data_tensor = data_tensor.unsqueeze(0).to(DEVICE)
    label_tensor = label_tensor.unsqueeze(0).to(DEVICE)
    min_distance=1.0
    max_distance=4.0
    path_loss_exponent=2.0
    reference_distance=1.0
    EPSILON = 0.1
    ALPHA = 0.01
    ATTACK_ITERATIONS = 40
    if min_distance >= 5.0:
        EPSILON = 0.6
        ALPHA = 0.021
        ATTACK_ITERATIONS = 100

    # EoT PGD attack
    x_adv = targeted_eot_pgd_attack(
        model=model,
        x=data_tensor,
        y=label_tensor,
        target_label=TARGET_LABEL,
        eps=EPSILON,
        alpha=ALPHA,
        num_iter=ATTACK_ITERATIONS,
        num_samples=10,
        min_distance=min_distance,
        max_distance=max_distance,
        path_loss_exponent=path_loss_exponent,
        reference_distance=reference_distance,
        noise_std=0.000001,
        k=20.0
    )

    # Save perturbation only
    original_np = data_tensor.squeeze().cpu().numpy()
    adv_np = x_adv.squeeze().detach().cpu().numpy()

    I_diff = adv_np[0] - original_np[0]
    Q_diff = adv_np[1] - original_np[1]

    interleaved = np.empty(I_diff.size + Q_diff.size, dtype=np.float32)
    interleaved[0::2] = I_diff
    interleaved[1::2] = Q_diff

    save_path = "May1/mid_dist_eot_pluto10_to_target1_noise.iq"
    interleaved.tofile(save_path)
    print(f"Saved EoT PGD noise to {save_path}")

generate_eot_pgd_noise()
