import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================
# 1. User-Customizable Parameters
# ==========================================
f_sig = 5.0          # Frequency of the original sine wave (Hz)
f_samp = 200.0       # Sampling frequency (Hz) - must be >= 2*f_sig
duration = 1.0       # Duration of the signal (seconds)
bits = 4             # Quantization bit depth
a1 = 0.95            # 1st-order predictor coefficient (close to 1 for high correlation)

# ==========================================
# 2. Signal Generation & Sampling
# ==========================================
# Time vector for the sampled discrete sequence x[n]
t = np.arange(0, duration, 1 / f_samp)

# High-resolution time vector to represent the "continuous" original analog wave x(t)
t_continuous = np.arange(0, duration, 1 / (f_samp * 10))
x_continuous = np.sin(2 * np.pi * f_sig * t_continuous)

# Sampled discrete signal x[n]
x_n = np.sin(2 * np.pi * f_sig * t)

# ==========================================
# 3. Quantizer Setup
# ==========================================
# To quantize the error efficiently, we estimate the maximum possible error boundary.
# The max amplitude is 1, so the maximum difference is roughly 2.
max_error = 2.0 
levels = 2 ** bits
delta = (2 * max_error) / levels  # Step size (Δ)

def quantize(val, delta):
    """Uniform scalar quantizer"""
    return np.round(val / delta) * delta

# ==========================================
# 4. Transmitter (Encoder) Loop
# ==========================================
e_q = np.zeros_like(x_n)      # Transmitted quantized error sequence e_q[n]
x_rec_tx = np.zeros_like(x_n) # Transmitter's internal reconstructed sequence x_rec[n]

for n in range(1, len(x_n)):
    # 1. Predict the upcoming sample
    x_hat = a1 * x_rec_tx[n-1]
    
    # 2. Calculate differential error
    e = x_n[n] - x_hat
    
    # 3. Quantize the error
    e_q[n] = quantize(e, delta)
    
    # 4. Reconstruct internally to prevent quantization noise accumulation
    x_rec_tx[n] = x_hat + e_q[n]

# -> The sequence e_q is what actually gets transmitted over the channel <-

# ==========================================
# 5. Receiver (Decoder) Loop
# ==========================================
x_rec_rx = np.zeros_like(e_q) # Receiver's reconstructed sequence

for n in range(1, len(e_q)):
    # Receiver uses the exact same predictor logic
    x_hat_rx = a1 * x_rec_rx[n-1]
    
    # Reconstruct discrete sample by adding received quantized error
    x_rec_rx[n] = x_hat_rx + e_q[n]

# ==========================================
# 6. Analog Reconstruction (Low-Pass Filter)
# ==========================================
# Instead of slow sinc interpolation, we use a Butterworth LPF 
# to smooth the discrete sequence back into a continuous analog waveform.
nyquist = 0.5 * f_samp
cutoff_freq = f_sig * 1.5  # Cutoff slightly above our signal frequency
b, a = signal.butter(4, cutoff_freq / nyquist, btype='low')

# Apply filter to the reconstructed discrete samples to get the analog wave
x_analog_rec = signal.filtfilt(b, a, x_rec_rx)

# Calculate MSE between discrete original and reconstructed
mse = np.mean((x_n - x_rec_rx)**2)

# ==========================================
# 7. Visualization
# ==========================================
plt.figure(figsize=(12, 8))

# Subplot 1: Transmitter tracking and original signal
plt.subplot(3, 1, 1)
plt.plot(t_continuous, x_continuous, 'k-', alpha=0.3, label="Original Analog $x(t)$")
plt.plot(t, x_n, 'ko', markersize=4, label="Sampled $x[n]$")
plt.plot(t, x_rec_tx, 'b-', label="Transmitter Tracking $x_{rec}[n]$")
plt.title("DPCM Encoder: Signal Tracking")
plt.legend(loc="upper right")
plt.grid(True)

# Subplot 2: Transmitted Prediction Error
plt.subplot(3, 1, 2)
plt.stem(t, e_q, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title(f"Quantized Prediction Error $e_q[n]$ ({bits}-bit resolution)")
plt.grid(True)

# Subplot 3: Receiver Analog Reconstruction
plt.subplot(3, 1, 3)
plt.plot(t_continuous, x_continuous, 'k--', label="Original Analog $x(t)$")
plt.plot(t, x_analog_rec, 'g-', linewidth=2, label="Reconstructed Analog $x_{rec}(t)$")
plt.title(f"Receiver Reconstruction (MSE = {mse:.6f})")
plt.legend(loc="upper right")
plt.grid(True)

plt.tight_layout()
plt.show()