import matplotlib.pyplot as plt
import numpy as np
import os
import glob

window_size = 20000
hop_size = 1
threshold_ratio = 0.75
sample_rate = 2e6  # 2 MHz

def load_iq_file(filepath):
    raw = np.fromfile(filepath, dtype=np.float32)
    iqdata = raw[::2] + 1j * raw[1::2]
    return iqdata

def detect_bluetooth_transmissions(iq_data):
    magnitudes = np.abs(iq_data)

    if len(magnitudes) < window_size:
        return []
    
    num_windows = (len(magnitudes) - window_size) // hop_size + 1
    avg_mags = np.array([
        np.mean(magnitudes[i * hop_size : i * hop_size + window_size])
        for i in range(num_windows)
    ])

    threshold = threshold_ratio * np.max(avg_mags)

    transmission_flags = avg_mags > threshold
    transmission_windows = []
    for i, flag in enumerate(transmission_flags):
        if flag:
            start = i * hop_size
            end = start + window_size
            transmission_windows.append((start, end))

    return transmission_windows

def merge_transmission_windows(windows):
    if not windows:
        return []

    merged = [windows[0]]
    for start, end in windows[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end: 
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged

def index_to_time(index):
    return index / sample_rate

if __name__ == "__main__":
    # Absolute path to the single .iq input file
    input_file = "/data2/brandan/RFSPOOF/Pluto_10_2m_run3.iq"

    # Absolute path for the output directory (can be different)
    output_folder = "/data2/brandan/RFSPOOF/nrf_out"
    os.makedirs(output_folder, exist_ok=True)

    # Verify input exists
    if not os.path.isfile(input_file):
        sys.stderr.write(f"Input file not found: {input_file}\n")
        sys.exit(1)

    # Load and detect
    iq_data = load_iq_file(input_file)
    windows = detect_bluetooth_transmissions(iq_data)
    merged_windows = merge_transmission_windows(windows)

    # Construct output filename
    base = os.path.basename(input_file)          # e.g. "Pluto_10_2m_run3.iq"
    name, _ = os.path.splitext(base)             # e.g. "Pluto_10_2m_run3"
    output_file = os.path.join(output_folder, f"{name}.txt")

    # Write results
    with open(output_file, "w") as f:
        for start, end in merged_windows:
            f.write(f"{index_to_time(start):.6f},{index_to_time(end):.6f}\n")

    print(f"Wrote {output_file}")