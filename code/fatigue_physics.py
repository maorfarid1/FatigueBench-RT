"""Physics utilities for FatigueBench-RT.

Basquin S-N with Goodman mean-stress correction, rainflow damage, Dirlik spectral damage,
Hermite (Winterstein) non-Gaussian transform and spectral-representation signal synthesis.
"""
import numpy as np
import rainflow
from scipy.signal import welch

FS = 512.0          # sampling frequency [Hz]
NW = 2048           # samples per monitoring window (4 s)
T_W = NW / FS       # window duration [s]
T_B = 3600.0        # duration of one operating block represented by a window [s]
SCALE = T_B / T_W   # extrapolation factor window -> block


def basquin_life(sa_eq, sf, b):
    """Cycles to failure N for equivalent fully reversed amplitude sa_eq (Basquin)."""
    sa_eq = np.maximum(sa_eq, 1e-9)
    return 0.5 * (sa_eq / sf) ** (1.0 / b)


def goodman(sa, sm, su):
    """Equivalent fully reversed amplitude (Goodman). Compressive means are not credited."""
    sm = np.maximum(sm, 0.0)
    ratio = np.clip(1.0 - sm / su, 1e-3, None)
    return sa / ratio


def rainflow_cycles(x):
    """Return amplitude, mean and count arrays of rainflow cycles (ASTM E1049 4-point)."""
    c = np.array([(r, m, n) for r, m, n, _, _ in rainflow.extract_cycles(x)])
    if c.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return c[:, 0] / 2.0, c[:, 1], c[:, 2]


def miner_damage(amp, mean, cnt, sf, b, su):
    """Palmgren-Miner damage of a cycle set."""
    n = basquin_life(goodman(amp, mean, su), sf, b)
    return np.sum(cnt / n)


def spectral_moments(x, fs=FS):
    f, p = welch(x - x.mean(), fs=fs, nperseg=512)
    m = [np.trapezoid(f ** i * p, f) for i in (0, 1, 2, 4)]
    return f, p, m


def dirlik_damage_rate(m, mean, sf, b, su, ngrid=400):
    """Dirlik (1985) damage per second for a stationary Gaussian process with moments m=(m0,m1,m2,m4)."""
    m0, m1, m2, m4 = m
    if m0 <= 0 or m2 <= 0 or m4 <= 0:
        return 0.0
    xm = m1 / m0 * np.sqrt(m2 / m4)
    g = m2 / np.sqrt(m0 * m4)
    d1 = 2.0 * (xm - g ** 2) / (1.0 + g ** 2)
    r = (g - xm - d1 ** 2) / (1.0 - g - d1 + d1 ** 2)
    d2 = (1.0 - g - d1 + d1 ** 2) / (1.0 - r)
    d3 = 1.0 - d1 - d2
    q = 1.25 * (g - d3 - d2 * r) / d1
    s = np.linspace(1e-6, 12.0, ngrid) * 2.0 * np.sqrt(m0)   # stress ranges
    z = s / (2.0 * np.sqrt(m0))
    pdf = (d1 / q * np.exp(-z / q) + d2 * z / r ** 2 * np.exp(-z ** 2 / (2 * r ** 2))
           + d3 * z * np.exp(-z ** 2 / 2.0)) / (2.0 * np.sqrt(m0))
    ep = np.sqrt(m4 / m2)
    n = basquin_life(goodman(s / 2.0, mean, su), sf, b)
    return ep * np.trapezoid(pdf / n, s)


def hermite_coeffs(skew, kurt):
    """Winterstein (1988) softening Hermite model coefficients (kurt > 3)."""
    if kurt <= 3.0 + 1e-6:
        return 0.0, 0.0, 1.0
    h4 = (np.sqrt(1.0 + 1.5 * (kurt - 3.0)) - 1.0) / 18.0
    h3 = skew / (6.0 * (1.0 + 6.0 * h4))
    k = 1.0 / np.sqrt(1.0 + 2.0 * h3 ** 2 + 6.0 * h4 ** 2)
    return h3, h4, k


def hermite_transform(u, skew, kurt):
    h3, h4, k = hermite_coeffs(skew, kurt)
    y = k * (u + h3 * (u ** 2 - 1.0) + h4 * (u ** 3 - 3.0 * u))
    return (y - y.mean()) / y.std()


def psd_shape(f, peaks):
    """Sum of Gaussian peaks; peaks = list of (fc, bw, weight)."""
    g = np.zeros_like(f)
    for fc, bw, w in peaks:
        g += w * np.exp(-0.5 * ((f - fc) / bw) ** 2)
    return g


def synth_gaussian(peaks, rng, n=NW, fs=FS):
    """Unit-variance stationary Gaussian process by spectral representation (random phases)."""
    f = np.fft.rfftfreq(n, 1.0 / fs)
    g = psd_shape(f, peaks)
    amp = np.sqrt(g) * np.exp(1j * rng.uniform(0, 2 * np.pi, f.size))
    amp[0] = 0.0
    x = np.fft.irfft(amp, n)
    return (x - x.mean()) / x.std()
