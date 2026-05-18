import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


def compute_spectrum(signal, dt=0.01, window="hann"):
    N = len(signal)

    if window == "hann":
        w = np.hanning(N)
    elif window == "hamming":
        w = np.hamming(N)
    else:
        w = np.ones(N)

    w = w / w.mean()
    spectrum = fft(signal * w)
    freqs = fftfreq(N, d=dt)

    mask = freqs > 0
    freqs = freqs[mask]
    power = (2.0 / N) * np.abs(spectrum[mask]) ** 2

    return freqs, power


def find_spectral_peaks(freqs, power, n=5, threshold=0.01):
    peaks, _ = find_peaks(power, height=power.max() * threshold, distance=5)
    order = np.argsort(power[peaks])[::-1]
    top = peaks[order[:n]]
    return freqs[top], power[top]


def frequency_error(f_numerical, f_theoretical):
    return abs(f_numerical - f_theoretical) / f_theoretical * 100
