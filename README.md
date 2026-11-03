# End-to-End DPCM Simulation in Python

**Subject:** Analog and Digital Communications
**Topic:** Differential Pulse Code Modulation (DPCM) — transmitter, channel and receiver simulated end to end.

---

## 1. Objective

To simulate a complete DPCM communication link in Python:

1. **Transmitter** – predict each sample, quantize the *prediction error*, and convert it to a bit stream.
2. **Channel** – ideal, binary symmetric channel (BSC), or BPSK over AWGN.
3. **Receiver** – recover the quantized error from the bits and rebuild the signal.

and to measure how much better DPCM performs than ordinary PCM at the same bit rate.

---

## 2. Theory in brief

Neighbouring samples of a band-limited signal (speech, audio) are strongly correlated. Instead of
coding `x[n]` itself, DPCM codes only the *unpredictable part* — the difference between the sample
and a prediction of it. That difference has a much smaller variance, so for the same number of bits
the quantization step is smaller and the SNR is higher.

### Transmitter (closed-loop)

```
                 +                         e[n]                     e_q[n]
 x[n] ---------> (+) ------------------> [Quantizer] ----+----------------> index -> b bits -> channel
                  ^ -                                    |
                  |                                      v
              x_pred[n]  <---- [Predictor] <---- x_rec[n] = x_pred[n] + e_q[n]
```

| Quantity | Equation |
|---|---|
| Prediction | `x_pred[n] = a1·x_rec[n-1] + a2·x_rec[n-2] + … + ap·x_rec[n-p]` |
| Prediction error | `e[n] = x[n] − x_pred[n]` |
| Quantized error | `e_q[n] = Q(e[n])` |
| Reconstruction | `x_rec[n] = x_pred[n] + e_q[n]` |

The predictor uses the **reconstructed** samples `x_rec` (not the original `x`). This is the key idea
of closed-loop DPCM: the receiver only ever has `x_rec`, so the transmitter must predict from the
same data. It also stops the quantization errors from accumulating — the overall error is
`x[n] − x_rec[n] = e[n] − e_q[n]`, i.e. just the quantizer error of that one sample.

### Receiver

```
 bits -> index -> e_q[n] ---> (+) ---> x_rec[n]  (output)
                               ^
                               |
                           x_pred[n] <---- [Predictor] <---- x_rec[n]
```

The receiver contains an identical predictor. On an error-free channel its output is identical to the
transmitter's local reconstruction (the program asserts this).

### Predictor design

The coefficients are the minimum mean-square-error linear predictor, obtained from the autocorrelation
`R(k)` by solving the Yule–Walker (Wiener–Hopf) equations `R·a = r`. They are computed once and are assumed
known to the receiver (sent as side information). The first `p` samples are sent unquantized as a start-up header.

### Quantizer

`b`-bit uniform **mid-rise** quantizer with `L = 2^b` levels over `[−vmax, +vmax]`, step size `Δ = 2·vmax / L`.
`vmax` is the peak of the prediction error. Each quantizer index is mapped to a `b`-bit natural binary word.

### Performance measures

* **Prediction gain** `Gp = σx² / σe²` (dB) – how much the predictor reduces the signal variance.
* **SQNR** `= Σx² / Σ(x − x̂)²` (dB).
* Ideal DPCM gain over PCM ≈ `Gp`.

### Channel models

* **BSC** – every bit is flipped independently with probability `p`.
* **BPSK/AWGN** – bits mapped to ±1, Gaussian noise added with `σ² = 1 / (2·Eb/N0)`, hard-decision detection.
  Theoretical BER = `½·erfc(√(Eb/N0))`; the simulation reproduces this curve.

---

## 3. Project structure

```
dpcm_project/
├── dpcm.py            # building blocks: signals, MP3/WAV loading, predictor, quantizer, Tx, channels, Rx, metrics
├── main.py            # runs the end-to-end simulation, sweeps and plots
├── requirements.txt   # numpy, matplotlib
├── README.md          # this file
└── results/           # generated plots, original.wav / received.wav, console output
```

---

## 4. How to run

```bash
pip install -r requirements.txt

python main.py                                   # speech-like signal, 3 bits, order-2 predictor
python main.py --signal sine --bits 4 --order 1  # different signal / parameters
python main.py --ber 0.001                       # add channel errors (BSC)
python main.py --ebn0 6                          # BPSK over AWGN at Eb/N0 = 6 dB
python main.py --signal wav --wav input.wav --seconds 2   # your own 16-bit WAV file
python main.py --no-sweeps                       # skip the parameter sweeps (faster)
```

### Running DPCM on an MP3 file and listening to the result

```bash
pip install miniaudio                            # one-time: MP3 decoder (see note below)

python main.py --signal mp3 --mp3 song.mp3 --bits 6 --order 4 --play
python main.py --signal mp3 --mp3 song.mp3 --start 30 --seconds 15 --bits 5 --order 4
```

What happens: the MP3 is decoded, the stereo channels are averaged to **mono**, and the chosen
excerpt (`--start`, `--seconds`; default first 10 s) goes through the full transmitter → channel → receiver chain
at the file's **original sampling rate**. Two WAV files are written to `results/`:

| File | Content |
|---|---|
| `original.wav` | the (mono) excerpt that went into the transmitter |
| `received.wav` | the output of the DPCM receiver — **this is what you listen to** |

Add `--play` to hear the original followed by the DPCM output straight away (uses `winsound` on Windows,
`afplay` on macOS, `paplay`/`aplay`/`ffplay` on Linux). If no player is found, just open the two WAV files
in any media player.

**MP3 decoding.** Python cannot decode MP3 on its own, so `dpcm.py` tries, in order: `miniaudio`
(`pip install miniaudio`, nothing else needed), `soundfile` (`pip install soundfile`) and the `ffmpeg`
command-line program (if it is installed and on the PATH). Any one of them is enough.

**What to try**
* `--bits 2 / 4 / 6 / 8` — hear the quantization noise fall as the bits increase (roughly 6 dB per bit).
* Compare against plain PCM at the same bits using the printed SQNR numbers.
* `--ber 1e-5`, `1e-4` — hear what channel errors do to the received audio.
* `--vmax-percentile 99.9` — a finer quantizer step for music with occasional loud transients.

| Option | Meaning | Default |
|---|---|---|
| `--signal` | `sine`, `multitone`, `speech`, `wav`, `mp3` | `speech` |
| `--wav` / `--mp3` | path to the input file (with `--signal wav` / `--signal mp3`) | – |
| `--start` | start position inside the WAV/MP3 file (seconds) | `0` |
| `--seconds` | duration to simulate | `1` (synthetic), `10` (files) |
| `--fs` | sampling rate for synthetic signals | `8000` |
| `--bits` | bits per sample | `3` |
| `--order` | predictor order `p` | `2` |
| `--vmax-percentile` | quantizer range = this percentile of `|e|` (100 = peak) | `100` |
| `--ber` | BSC bit-error probability | `0` |
| `--ebn0` | BPSK/AWGN Eb/N0 in dB (overrides `--ber`) | – |
| `--play` | play original, then DPCM output | off |
| `--out` | output folder | `results` |
| `--no-sweeps` | skip the parameter sweeps | off |

For long recordings the parameter sweeps (plots 03–06) are run on a short excerpt (about 0.5 s) taken from the
middle of the signal so that they finish quickly; the main run and the WAV files use the full selected excerpt.

---

## 5. Results (default run: speech-like signal, 8 kHz, 3 bits/sample, order 2)

| Metric | Value |
|---|---|
| Bit rate | 24 kbit/s |
| Predictor coefficients | a1 = 1.7275, a2 = −0.7875 |
| Prediction gain | 14.02 dB |
| **DPCM SQNR** | **24.03 dB** |
| PCM SQNR (same 3 bits) | 15.11 dB |
| **DPCM advantage** | **≈ 8.9 dB** |

SQNR versus number of bits (error-free channel):

| bits | DPCM (dB) | PCM (dB) |
|---|---|---|
| 1 | 10.29 | 3.51 |
| 2 | 17.82 | 8.91 |
| 3 | 24.03 | 15.11 |
| 4 | 30.08 | 21.21 |
| 5 | 36.07 | 26.95 |
| 6 | 42.11 | 32.98 |
| 7 | 48.20 | 39.02 |
| 8 | 54.19 | 45.07 |

Plots in `results/`:

| File | Shows |
|---|---|
| `01_waveforms.png` | original vs received signal, prediction error vs quantized error, reconstruction error |
| `02_histograms.png` | amplitude histograms of `x[n]` and `e[n]` (prediction narrows the distribution) |
| `03_snr_vs_bits.png` | DPCM vs PCM SQNR as bits per sample increase |
| `04_snr_vs_order.png` | effect of predictor order |
| `05_snr_vs_ber.png` | end-to-end SNR vs channel bit error rate |
| `06_link_vs_ebn0.png` | simulated vs theoretical BPSK BER, and DPCM SNR vs Eb/N0 |

`original.wav` and `received.wav` can be played to hear the result.

### Observations

* DPCM beats PCM by roughly the prediction gain (≈ 9 dB here, i.e. about 1.5 extra bits' worth of quality) at the same bit rate; the same quality as 3-bit PCM needs only ≈ 2 bits with DPCM.
* The biggest improvement comes from going from order 1 to order 2; higher orders give diminishing returns.
* Each extra bit adds ≈ 6 dB, the usual `6.02·b` rule.
* DPCM is sensitive to channel errors: a wrong bit corrupts `e_q[n]` and, because the predictor feeds back
  its own output, the error spreads to following samples (it decays because the predictor is stable).
  A single most-significant-bit error causes a large error. In the noisy-channel runs the end-to-end SNR drops sharply
  compared to the error-free value (e.g. ≈ 24 dB → ≈ 4.7 dB at BER ≈ 1 %).
* **Predictor order trades quality for robustness.** On a synthetic music-like test clip (6 bits, BER = 10⁻⁴) the end-to-end SNR was
  about 10.7 dB with an order-1 predictor but only about 6.5 dB with order 4, even though order 4 gives the higher
  SNR on a clean channel. A stronger predictor has poles closer to the unit circle, so a wrong bit "rings" for longer in the
  receiver. This is why practical DPCM/ADPCM systems use leaky predictors or error protection.
* **DPCM is not a compression scheme like MP3.** DPCM still sends `b` bits for every sample (6 bits × 44.1 kHz ≈ 265 kbit/s
  per channel), whereas a typical MP3 uses about 128 kbit/s for both channels. DPCM's benefit is better quality than PCM at the same
  bit rate, or the same quality with fewer bits per sample.

---

## 6. Assumptions and limitations

* Predictor coefficients and `vmax` are known at the receiver (side information); the first `p` samples are sent without quantization.
* Fixed (non-adaptive) predictor and quantizer — no adaptive DPCM (ADPCM).
* Natural binary mapping of quantizer indices; no channel coding or Gray coding.
* The "speech-like" test signal is synthetic; use `--signal wav` or `--signal mp3` for real audio.
* Audio files are converted to mono (channel average) before processing; the sampling rate is not changed.
* The whole excerpt is processed in memory with a per-sample Python loop (about 0.5 s per 5 s of CD-quality audio); keep `--seconds` to a minute or two.

## 7. Possible extensions

* Adaptive quantizer / predictor (ADPCM).
* Gray-coded indices, or error-protection codes to improve robustness to bit errors.
* Non-uniform (µ-law) quantizer for the prediction error.
* Compare with delta modulation (1-bit, order-1 DPCM).
