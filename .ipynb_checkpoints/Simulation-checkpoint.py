import numpy as np
from scipy.signal import kaiserord, firwin, lfilter
import gzip
import pickle
import scipy.signal as signal
import math
import torch

### Transmitter Data ###
# Parameters
FILENAME_TX = f"/home/jfeng/Desktop/jfeng/rf_spoofing/spoofing/1m_2m_replacedPluto4/Pluto_10_windows_runs2_3/Pluto_10_2m_run3.iq"
print("Current Transmitter File Processing: ", FILENAME_TX)
# Read in the data
data_tx = np.fromfile(FILENAME_TX, dtype="float32")
# Bring I and Q into one value
I = data_tx[0::2]
Q = data_tx[1::2]
data_complex_tx = I + 1j*Q

### Adversary Noise Data ###
#FILENAME_ADV_TX = f"/home/jfeng/Desktop/jfeng/rf_spoofing/May1/test2.iq"
FILENAME_ADV_TX = f"/data2/brandan/RFSPOOF/May16/pluto10_to_target1_noise.iq" 
print("Current Adversary Noise File Processing: ", FILENAME_ADV_TX)
# Read in the data
data_adv_tx = np.fromfile(FILENAME_ADV_TX, dtype="float32")
# Bring I and Q into one value
I = data_adv_tx[0::2]
Q = data_adv_tx[1::2]
data_complex_adv_tx = I + 1j*Q

#signal_power = np.mean(np.abs(data_tx)**2)
#noise_std = np.sqrt(signal_power * 0.01)
#print("Noise std: ", noise_std)

# Adjust
# Path loss exponent
n = 2
# Reference distance
d1 = 1 
# Actual distance
d2 = 4
# Noise
#noise_std = 0.000001
#data_complex_adv_tx = torch.from_numpy(data_complex_adv_tx).to(torch.cfloat)
# Path loss
scaling = (d1 / d2) ** (n / 2)
iq_path = data_complex_adv_tx * scaling
print("Scaled signal: ", iq_path)
# Rician fading
k = 10.0
h_nlos = (np.random.randn(*iq_path.shape) + 1j * np.random.randn(*iq_path.shape)) / np.sqrt(2)
h_los = np.ones_like(iq_path, dtype=iq_path.dtype)
scaling_los = np.sqrt(k / (k + 1))
scaling_nlos = np.sqrt(1 / (k + 1))
h_rician = scaling_los * h_los + scaling_nlos * h_nlos
iq_scaled = iq_path * h_rician


signal_power = np.mean(np.abs(iq_scaled)**2)
noise_std = np.sqrt(signal_power * 0.01)
print("Noise std: ", noise_std)

#fading = (torch.randn_like(iq_scaled) + 1j * torch.randn_like(iq_scaled)) / np.sqrt(2)
#iq_faded = iq_scaled * fading
#print ("Faded signal: ", iq_faded)

# AWGN noise
noise = noise_std * (np.random.randn(*iq_scaled.shape) + 1j * np.random.randn(*iq_scaled.shape)) / np.sqrt(2)
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
data_to_save = [elem for pair in zip(real, imag) for elem in pair]
data_to_save = np.array(data_to_save)
filtered_signal_float32 = data_to_save.astype("float32")
filtered_signal_float32.tofile("May16/test_sample_test1.iq")

