"""
dpcm.py - Core building blocks of an end-to-end DPCM communication system.

    x[n] --> [ TRANSMITTER ] --> bits --> [ CHANNEL ] --> bits' --> [ RECEIVER ] --> x_hat[n]

Transmitter (closed-loop DPCM encoder)
    prediction   : x_pred[n] = sum_k a_k * x_rec[n-k]     (uses RECONSTRUCTED samples)
    error        : e[n]      = x[n] - x_pred[n]
    quantizer    : e_q[n]    = Q(e[n])                    (b-bit uniform mid-rise)
    local decode : x_rec[n]  = x_pred[n] + e_q[n]
    line coding  : index of e_q[n] -> b-bit binary word

Channel
    Binary symmetric channel (flips bits with probability p), or
    BPSK over AWGN with hard-decision detection (given Eb/N0).

Receiver (DPCM decoder)
    bits -> indices -> e_q[n] -> x_rec[n] = x_pred[n] + e_q[n]
    (same predictor as the transmitter, driven by its own reconstructed output)
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Signal sources
# --------------------------------------------------------------------------- #
def make_signal(kind: str = "speech", fs: int = 8000, duration: float = 1.0,
                seed: int = 0) -> np.ndarray:
    """Generate a test signal normalised to peak amplitude 1.

    kind = "sine"       : single 300 Hz tone
          "multitone"   : 200 + 550 + 1100 Hz
          "speech"      : speech-like (harmonic voiced sound, slowly varying
                          pitch and amplitude envelope, plus a little noise)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration)) / fs

    if kind == "sine":
        x = np.sin(2 * np.pi * 300 * t)
    elif kind == "multitone":
        x = (np.sin(2 * np.pi * 200 * t)
             + 0.6 * np.sin(2 * np.pi * 550 * t)
             + 0.3 * np.sin(2 * np.pi * 1100 * t))
    elif kind == "speech":
        f0 = 120 + 20 * np.sin(2 * np.pi * 1.5 * t)           # wandering pitch
        phase = 2 * np.pi * np.cumsum(f0) / fs
        x = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 12))
        envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 3 * t) ** 2   # syllable rate
        x = x * envelope + 0.02 * rng.standard_normal(t.size)
    else:
        raise ValueError(f"unknown signal kind: {kind!r}")

    return x / np.max(np.abs(x))


def _to_mono_normalised(data: np.ndarray) -> np.ndarray:
    """(N, channels) array -> mono (channel average) normalised to peak 1."""
    x = np.asarray(data, dtype=float).reshape(len(data), -1).mean(axis=1)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak == 0:
        raise ValueError("The selected part of the audio file is empty or silent.")
    return x / peak


def load_wav(path: str, max_seconds: float | None = None,
             start_seconds: float = 0.0) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file (channels averaged to mono), normalised to [-1, 1]."""
    with wave.open(path, "rb") as w:
        fs, n_ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise ValueError("Only 16-bit PCM WAV files are supported.")
        w.setpos(min(int(start_seconds * fs), w.getnframes()))
        left = w.getnframes() - w.tell()
        n = left if max_seconds is None else min(left, int(max_seconds * fs))
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype="<i2").astype(float).reshape(-1, n_ch)
    return _to_mono_normalised(data), fs


# ---- MP3 / general audio decoding ------------------------------------------
# Python's standard library cannot decode MP3, so we try, in order:
#   1. miniaudio   (pip install miniaudio   - self-contained, nothing else needed)
#   2. soundfile   (pip install soundfile   - needs libsndfile >= 1.1 for MP3)
#   3. ffmpeg      (command-line program installed on the system PATH)
def _decode_miniaudio(path, start, seconds):
    import miniaudio
    info = miniaudio.get_file_info(path)
    dec = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.FLOAT32,
                                nchannels=info.nchannels, sample_rate=info.sample_rate)
    fs = dec.sample_rate
    data = np.frombuffer(dec.samples, dtype=np.float32).reshape(-1, dec.nchannels)
    a = int(start * fs)
    b = None if seconds is None else a + int(seconds * fs)
    return _to_mono_normalised(data[a:b]), fs


def _decode_soundfile(path, start, seconds):
    import soundfile as sf
    fs = sf.info(path).samplerate
    frames = -1 if seconds is None else int(seconds * fs)
    data, _ = sf.read(path, start=int(start * fs), frames=frames,
                      dtype="float64", always_2d=True)
    return _to_mono_normalised(data), fs


def _decode_ffmpeg(path, start, seconds):
    import os
    import shutil
    import subprocess
    import tempfile
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
        if start > 0:
            cmd += ["-ss", str(start)]
        cmd += ["-i", path]
        if seconds is not None:
            cmd += ["-t", str(seconds)]
        cmd += ["-ac", "1", "-acodec", "pcm_s16le", tmp]       # mono, native sample rate
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode(errors="replace").strip().splitlines()
            raise RuntimeError(detail[-1] if detail else "ffmpeg could not read the file")
        return load_wav(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_audio(path: str, seconds: float | None = None,
               start: float = 0.0) -> tuple[np.ndarray, int]:
    """Load an audio file (MP3, or WAV/FLAC/OGG where the decoder allows).

    Returns (mono signal normalised to peak 1, sampling rate in Hz).
    Only `seconds` of audio beginning at `start` seconds are decoded.
    """
    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.lower().endswith(".wav"):
        try:
            return load_wav(path, seconds, start)
        except Exception:
            pass                                    # e.g. 24-bit WAV -> try the decoders

    problems = []
    for name, fn in (("miniaudio", _decode_miniaudio),
                     ("soundfile", _decode_soundfile),
                     ("ffmpeg", _decode_ffmpeg)):
        try:
            return fn(path, start, seconds)
        except ImportError:
            problems.append(f"  - {name}: not installed")
        except Exception as exc:                    # decoder present but failed
            problems.append(f"  - {name}: {exc}")
    raise RuntimeError(
        "Could not decode the audio file. If no decoder is installed, run\n"
        "    pip install miniaudio\n"
        "(or install ffmpeg and add it to PATH). Otherwise the file may be\n"
        "corrupt or not an audio file. Details:\n" + "\n".join(problems))


def save_wav(path: str, x: np.ndarray, fs: int) -> None:
    """Write a signal in [-1, 1] to a 16-bit mono WAV file."""
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- #
# 2. Predictor design
# --------------------------------------------------------------------------- #
def design_predictor(x: np.ndarray, order: int) -> np.ndarray:
    """Optimal (minimum mean-square error) linear predictor of a given order.

    Solves the Wiener-Hopf / Yule-Walker normal equations  R a = r  where
    R is the Toeplitz autocorrelation matrix and r = [R(1) ... R(p)].
    """
    if order == 0:
        return np.zeros(0)
    N = len(x)
    r = np.array([np.dot(x[: N - k], x[k:]) / N for k in range(order + 1)])
    R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
    R += 1e-9 * np.eye(order)                      # numerical safety
    return np.linalg.solve(R, r[1: order + 1])


# --------------------------------------------------------------------------- #
# 3. Quantizer
# --------------------------------------------------------------------------- #
class UniformQuantizer:
    """b-bit uniform MID-RISE quantizer on [-vmax, +vmax].

    L = 2^b levels, step size  Delta = 2*vmax / L.
    Inputs outside the range are clipped (overload).
    """

    def __init__(self, bits: int, vmax: float):
        self.bits = bits
        self.levels = 2 ** bits
        self.vmax = vmax
        self.step = 2.0 * vmax / self.levels

    def encode(self, v):
        idx = np.floor(np.asarray(v) / self.step) + self.levels // 2
        return np.clip(idx, 0, self.levels - 1).astype(int)

    def decode(self, idx):
        return (np.asarray(idx) - self.levels // 2 + 0.5) * self.step


# --------------------------------------------------------------------------- #
# 4. Bit <-> index conversion (line coding)
# --------------------------------------------------------------------------- #
def indices_to_bits(idx: np.ndarray, bits: int) -> np.ndarray:
    shifts = np.arange(bits - 1, -1, -1)
    return ((np.asarray(idx)[:, None] >> shifts) & 1).astype(np.uint8).ravel()


def bits_to_indices(stream: np.ndarray, bits: int) -> np.ndarray:
    weights = 1 << np.arange(bits - 1, -1, -1)
    return stream.reshape(-1, bits).astype(int) @ weights


# --------------------------------------------------------------------------- #
# 5. Transmitter
# --------------------------------------------------------------------------- #
@dataclass
class DPCMParams:
    """Side information both ends agree on (sent once, not per sample)."""
    bits: int
    order: int
    coeffs: np.ndarray
    vmax: float
    header: np.ndarray                 # first `order` samples, sent as raw PCM


@dataclass
class TxResult:
    indices: np.ndarray                # quantizer indices for n >= order
    bitstream: np.ndarray              # serial bits sent to the channel
    params: DPCMParams
    prediction: np.ndarray             # x_pred[n]
    error: np.ndarray                  # e[n]   (prediction error, input to Q)
    error_q: np.ndarray                # e_q[n] (quantized prediction error)
    x_rec: np.ndarray                  # transmitter's local reconstruction
    traces: dict = field(default_factory=dict)


def dpcm_transmit(x: np.ndarray, bits: int = 3, order: int = 2,
                  coeffs: np.ndarray | None = None,
                  vmax: float | None = None,
                  vmax_percentile: float = 100.0) -> TxResult:
    """Closed-loop DPCM encoder.

    vmax_percentile < 100 sets the quantizer range from that percentile of |e|
    instead of the peak, giving a finer step (rare large errors are clipped).
    Useful for music, where a few transients would otherwise dominate the range.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    a = design_predictor(x, order) if coeffs is None else np.asarray(coeffs, float)

    if vmax is None:   # range of the open-loop prediction error (vectorised)
        if order:
            pred_ol = np.convolve(x, np.concatenate(([0.0], a)))[:N]
        else:
            pred_ol = np.zeros(N)
        vmax = float(np.percentile(np.abs(x[order:] - pred_ol[order:]), vmax_percentile))
        vmax = max(vmax, 1e-12)

    q = UniformQuantizer(bits, vmax)
    x_rec = np.zeros(N)
    x_rec[:order] = x[:order]                       # header (lossless warm-up)
    pred = np.zeros(N)
    err = np.zeros(N)
    err_q = np.zeros(N)
    idx = np.zeros(N - order, dtype=int)
    a_rev = a[::-1].copy()                          # so that a_rev . x_rec[n-p:n] = sum a_k x_rec[n-k]
    half, top, step = q.levels // 2, q.levels - 1, q.step

    for n in range(order, N):
        pr = float(np.dot(a_rev, x_rec[n - order:n])) if order else 0.0
        e = x[n] - pr
        k = int(np.floor(e / step)) + half          # quantizer index (clipped to the valid range)
        k = 0 if k < 0 else top if k > top else k
        eq = (k - half + 0.5) * step                # de-quantized error
        pred[n], err[n], err_q[n], idx[n - order] = pr, e, eq, k
        x_rec[n] = pr + eq                          # feedback loop

    params = DPCMParams(bits, order, a, vmax, x[:order].copy())
    return TxResult(idx, indices_to_bits(idx, bits), params, pred, err, err_q, x_rec)


# --------------------------------------------------------------------------- #
# 6. Channels
# --------------------------------------------------------------------------- #
def binary_symmetric_channel(stream: np.ndarray, ber: float,
                             rng: np.random.Generator | None = None) -> np.ndarray:
    """Flip each bit independently with probability `ber`."""
    rng = rng or np.random.default_rng()
    flips = rng.random(stream.shape) < ber
    return np.bitwise_xor(stream, flips.astype(np.uint8))


def bpsk_awgn_channel(stream: np.ndarray, ebn0_db: float,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """BPSK (0 -> -1, 1 -> +1) over AWGN, hard-decision detection.

    With Eb = 1:  sigma^2 = N0/2 = 1 / (2 * Eb/N0).
    Theoretical BER = 0.5 * erfc( sqrt(Eb/N0) ).
    """
    rng = rng or np.random.default_rng()
    sigma = math.sqrt(1.0 / (2.0 * 10 ** (ebn0_db / 10.0)))
    tx = 2.0 * stream.astype(float) - 1.0
    rx = tx + sigma * rng.standard_normal(tx.shape)
    return (rx > 0).astype(np.uint8)


def theoretical_bpsk_ber(ebn0_db: float) -> float:
    return 0.5 * math.erfc(math.sqrt(10 ** (ebn0_db / 10.0)))


# --------------------------------------------------------------------------- #
# 7. Receiver
# --------------------------------------------------------------------------- #
def dpcm_receive(rx_bits: np.ndarray, params: DPCMParams) -> np.ndarray:
    """DPCM decoder: bits -> indices -> e_q -> x_rec = x_pred + e_q."""
    idx = bits_to_indices(rx_bits, params.bits)
    q = UniformQuantizer(params.bits, params.vmax)
    p = params.order
    a_rev = params.coeffs[::-1].copy()
    N = len(idx) + p
    e_q = q.decode(idx)                              # whole array at once

    x_out = np.zeros(N)
    x_out[:p] = params.header
    for n in range(p, N):
        pr = float(np.dot(a_rev, x_out[n - p:n])) if p else 0.0
        x_out[n] = pr + e_q[n - p]
    return x_out


# --------------------------------------------------------------------------- #
# 8. PCM baseline & metrics
# --------------------------------------------------------------------------- #
def pcm_roundtrip(x: np.ndarray, bits: int) -> np.ndarray:
    """Plain PCM (quantise the signal itself) with the same number of bits."""
    q = UniformQuantizer(bits, float(np.max(np.abs(x))))
    return q.decode(q.encode(x))


def snr_db(reference: np.ndarray, test: np.ndarray, skip: int = 0) -> float:
    ref, tst = reference[skip:], test[skip:]
    noise = np.sum((ref - tst) ** 2)
    if noise == 0:
        return float("inf")
    return 10 * math.log10(np.sum(ref ** 2) / noise)


def prediction_gain_db(x: np.ndarray, e: np.ndarray, skip: int = 0) -> float:
    return 10 * math.log10(np.var(x[skip:]) / np.var(e[skip:]))
