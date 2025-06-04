import numpy as np
from scipy.signal import kaiserord, firwin, lfilter
import gzip
import pickle
import scipy.signal as signal
import math
import torch

def apply_iq_imbalances(x, G_I, G_Q, phi, dc_I, dc_Q):
    I = x.real
    Q = x.imag
    ratio = G_Q / G_I
    G_I_ratio = np.sqrt(2 / (1 + ratio**2))
    G_Q_ratio = ratio * G_I
    I_g = G_I_ratio * I
    Q_g = G_Q_ratio * Q

    '''
    I_g = G_I * I
    Q_g = G_Q * Q
    '''
    cos_phi = np.cos(phi)
    tan_phi = np.tan(phi)
    Q_phi = Q_g / cos_phi + I_g * tan_phi
    I_phi = I_g
    I_final = I_phi + dc_I
    Q_final = Q_phi + dc_Q
    x_out = I_final + Q_final * 1j
    return x_out

### Transmitter Data ###
FILENAME_TX = f"/home/jfeng/Desktop/May27/Pluto_10_test.iq"

print("Current Transmitter File Processing: ", FILENAME_TX)
# Read in the data
data_tx = np.fromfile(FILENAME_TX, dtype="float32")
# Bring I and Q into one value
I = data_tx[0::2]
Q = data_tx[1::2]
data_complex_tx = I + 1j*Q

FILENAME_ADV_TX = f"/data2/brandan/RFSPOOF/May27/goodParams.iq"

print("Current Adversary Noise File Processing: ", FILENAME_ADV_TX)
# Read in the data
data_adv_tx = np.fromfile(FILENAME_ADV_TX, dtype="float32")
# Bring I and Q into one value
I = data_adv_tx[0::2]
Q = data_adv_tx[1::2]
data_complex_adv_tx = I + 1j*Q

data_complex_adv_tx = data_complex_adv_tx
for i in range (0, len(data_complex_adv_tx)):
    if (abs(data_complex_adv_tx[i]) > 1):
        print("oops")


n = 2
d1 = 1 
d2 = 2 

x_imbalance = apply_iq_imbalances(data_complex_adv_tx, 0.04, 0.04, -0.0004, -0.00005, -0.000005) 

scaling = (d1 / d2) ** (n / 2)
iq_path = x_imbalance * scaling * 0.05
print("Scaled signal: ", iq_path)
k = 20.0

i_window = iq_path.real
q_window = iq_path.imag
scale_LOS = np.sqrt(k / (k + 1))
scale_NLOS = np.sqrt(1 / (k + 1))
real_nlos = np.random.randn(*i_window.shape) / np.sqrt(2.0)
imag_nlos = np.random.randn(*i_window.shape) / np.sqrt(2.0)
real_los = np.ones(i_window.shape)
imag_los = np.zeros(i_window.shape)
real_fade = scale_LOS * real_los + scale_NLOS * real_nlos
imag_fade = scale_LOS * imag_los + scale_NLOS * imag_nlos
faded_real = i_window * real_fade - q_window * imag_fade
faded_imag = i_window * imag_fade + q_window * real_fade
iq_scaled = faded_real + 1j*faded_imag

noise_std = 0.000001

# AWGN noise
per_dim_std = noise_std / np.sqrt(2.0)
noise = np.random.randn(*iq_scaled.shape) * per_dim_std
iq_received = iq_scaled + noise

### Combine Datas ###
len_data_complex_tx = len(data_complex_tx)
print("Length of transmitter: ", len_data_complex_tx)
len_data_complex_adv_tx = len(data_complex_adv_tx)
print("Length of noise: ", len_data_complex_adv_tx)
num_chunks = len_data_complex_tx // len_data_complex_adv_tx
combined_array = data_complex_tx.copy()
for i in range(num_chunks):
    start = i * len_data_complex_adv_tx
    end = start + len_data_complex_adv_tx
    combined_array[start:end] = data_complex_tx[start:end] + iq_received

print("Original Tx: ", data_complex_tx)
print("Noise: ", data_complex_adv_tx)
print("Path loss noise: ", iq_path)
print("Rician noise: ", iq_scaled)
print("With noise: ", iq_received)
print("Combined: ", combined_array)
real = combined_array.real
imag = combined_array.imag
print("Mean Magnitude tx: ", np.mean(np.abs(data_complex_tx)))
print("Mean Magnitude Noise: ", np.mean(np.abs(data_complex_adv_tx)))
print("Mean Magnitude after imbalance: ", np.mean(np.abs(x_imbalance)))
print("Mean Magnitude noise after transformations: ", np.mean(np.abs(iq_received)))
print("Mean Magnitude combined: ", np.mean(np.abs(real + imag * 1j)))
data_to_save = [elem for pair in zip(real, imag) for elem in pair]
data_to_save = np.array(data_to_save)
filtered_signal_float32 = data_to_save.astype("float32")
filtered_signal_float32.tofile("iq/May23/hewa.iq")

