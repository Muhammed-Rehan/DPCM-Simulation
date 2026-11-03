"""
main.py - Run the end-to-end DPCM simulation and generate all plots/results.

Examples
--------
    python main.py                                  # defaults: speech-like signal, 3 bits, order 2
    python main.py --signal sine --bits 4 --order 1
    python main.py --signal wav --wav input.wav --seconds 2
    python main.py --signal mp3 --mp3 song.mp3 --seconds 10 --bits 6 --play
    python main.py --ber 0.001                      # add a noisy channel (BSC)
    python main.py --ebn0 6                         # BPSK over AWGN at Eb/N0 = 6 dB
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")                       # works without a display
import matplotlib.pyplot as plt
import numpy as np

import dpcm as D


# --------------------------------------------------------------------------- #
# One complete transmitter -> channel -> receiver run
# --------------------------------------------------------------------------- #
def run_link(x, bits, order, ber=0.0, ebn0=None, seed=1, vmax_percentile=100.0):
    rng = np.random.default_rng(seed)
    tx = D.dpcm_transmit(x, bits=bits, order=order, vmax_percentile=vmax_percentile)

    if ebn0 is not None:
        rx_bits = D.bpsk_awgn_channel(tx.bitstream, ebn0, rng)
    elif ber > 0:
        rx_bits = D.binary_symmetric_channel(tx.bitstream, ber, rng)
    else:
        rx_bits = tx.bitstream.copy()

    x_hat = D.dpcm_receive(rx_bits, tx.params)
    bit_errors = int(np.sum(rx_bits != tx.bitstream))
    return tx, rx_bits, x_hat, bit_errors


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_waveforms(x, tx, x_hat, fs, order, outdir, ntaps=200, start=None):
    t = np.arange(len(x)) / fs
    start = order + 400 if start is None else start
    start = max(0, min(start, len(x) - ntaps))
    s = slice(start, start + ntaps)                       # a short zoomed window
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(t[s], x[s], label="original x[n]")
    ax[0].step(t[s], x_hat[s], where="mid", label="received x^[n]", alpha=0.8)
    ax[0].set_title("Original vs reconstructed signal (zoomed)")
    ax[0].legend(loc="upper right")

    ax[1].plot(t[s], tx.error[s], label="prediction error e[n]")
    ax[1].step(t[s], tx.error_q[s], where="mid", label="quantized error e_q[n]", alpha=0.8)
    ax[1].set_title("Prediction error before / after the quantizer")
    ax[1].legend(loc="upper right")

    ax[2].plot(t[s], x[s] - x_hat[s], color="crimson")
    ax[2].set_title("Reconstruction error x[n] - x^[n]")
    ax[2].set_xlabel("time (s)")

    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_waveforms.png"), dpi=150)
    plt.close(fig)


def plot_histograms(x, tx, order, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(x[order:], bins=80, color="steelblue")
    ax[0].set_title(f"Histogram of x[n]  (var = {np.var(x):.4f})")
    ax[1].hist(tx.error[order:], bins=80, color="darkorange")
    ax[1].set_title(f"Histogram of e[n]  (var = {np.var(tx.error[order:]):.5f})")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Prediction concentrates the amplitude distribution around zero")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_histograms.png"), dpi=150)
    plt.close(fig)


def plot_snr_vs_bits(x, order, outdir, max_bits=8):
    bits_list = np.arange(1, max_bits + 1)
    s_dpcm, s_pcm = [], []
    for b in bits_list:
        tx = D.dpcm_transmit(x, bits=int(b), order=order)
        s_dpcm.append(D.snr_db(x, tx.x_rec, skip=order))
        s_pcm.append(D.snr_db(x, D.pcm_roundtrip(x, int(b))))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bits_list, s_dpcm, "o-", label=f"DPCM (order {order})")
    ax.plot(bits_list, s_pcm, "s--", label="PCM")
    ax.set_xlabel("bits per sample")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("Quantization SNR vs bits per sample (error-free channel)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "03_snr_vs_bits.png"), dpi=150)
    plt.close(fig)
    return bits_list, np.array(s_dpcm), np.array(s_pcm)


def plot_snr_vs_order(x, bits, outdir, max_order=8):
    orders = np.arange(1, max_order + 1)
    gains, snrs = [], []
    for p in orders:
        tx = D.dpcm_transmit(x, bits=bits, order=int(p))
        gains.append(D.prediction_gain_db(x, tx.error, skip=int(p)))
        snrs.append(D.snr_db(x, tx.x_rec, skip=int(p)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(orders, gains, "o-", label="prediction gain (var x / var e)")
    ax.plot(orders, snrs, "s-", label=f"SQNR of DPCM ({bits} bits)")
    ax.set_xlabel("predictor order p")
    ax.set_ylabel("dB")
    ax.set_title("Effect of predictor order")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "04_snr_vs_order.png"), dpi=150)
    plt.close(fig)


def plot_snr_vs_ber(x, bits, order, outdir, trials=5):
    bers = np.logspace(-5, -1, 9)
    means = []
    for p in bers:
        vals = [D.snr_db(x, run_link(x, bits, order, ber=p, seed=s)[2], skip=order)
                for s in range(trials)]
        means.append(np.mean(vals))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(bers, means, "o-")
    ax.set_xlabel("channel bit error rate (BSC)")
    ax.set_ylabel("end-to-end SNR (dB)")
    ax.set_title("Effect of channel errors on the received signal")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "05_snr_vs_ber.png"), dpi=150)
    plt.close(fig)


def plot_snr_vs_ebn0(x, bits, order, outdir, trials=5):
    ebn0s = np.arange(0, 11, 1)
    means, ber_meas, ber_theory = [], [], []
    for e in ebn0s:
        snrs, errs = [], []
        for s in range(trials):
            tx, rx_bits, x_hat, ne = run_link(x, bits, order, ebn0=float(e), seed=s)
            snrs.append(D.snr_db(x, x_hat, skip=order))
            errs.append(ne / len(rx_bits))
        means.append(np.mean(snrs))
        ber_meas.append(np.mean(errs))
        ber_theory.append(D.theoretical_bpsk_ber(float(e)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].semilogy(ebn0s, ber_meas, "o", label="simulated")
    ax[0].semilogy(ebn0s, ber_theory, "-", label="theory: 0.5 erfc(sqrt(Eb/N0))")
    ax[0].set_xlabel("Eb/N0 (dB)")
    ax[0].set_ylabel("BER")
    ax[0].set_title("BPSK over AWGN")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend()
    ax[1].plot(ebn0s, means, "o-")
    ax[1].set_xlabel("Eb/N0 (dB)")
    ax[1].set_ylabel("end-to-end SNR (dB)")
    ax[1].set_title("DPCM link quality vs channel Eb/N0")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "06_link_vs_ebn0.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Playback (standard library / system players only)
# --------------------------------------------------------------------------- #
def play_wav(path):
    """Play a WAV file and wait until it finishes. Returns True on success."""
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
            return True
        candidates = (["afplay"] if sys.platform == "darwin"
                      else ["paplay", "aplay", "ffplay"])
        for prog in candidates:
            if shutil.which(prog):
                cmd = [prog, path]
                if prog == "ffplay":
                    cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]
                subprocess.run(cmd, check=True)
                return True
    except Exception as exc:
        print(f" [play] failed: {exc}")
        return False
    print(" [play] no audio player found - open the WAV files in the results folder instead.")
    return False


# --------------------------------------------------------------------------- #
SWEEP_MAX_SAMPLES = 24000          # parameter sweeps use at most this many samples (keeps them quick)


def main():
    ap = argparse.ArgumentParser(description="End-to-end DPCM simulation")
    ap.add_argument("--signal", default="speech",
                    choices=["sine", "multitone", "speech", "wav", "mp3"])
    ap.add_argument("--wav", help="path to a 16-bit PCM WAV file (with --signal wav)")
    ap.add_argument("--mp3", help="path to an MP3 file (with --signal mp3)")
    ap.add_argument("--start", type=float, default=0.0,
                    help="where to start in the WAV/MP3 file, in seconds (default 0)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="duration to simulate (default: 1 s synthetic, 10 s for WAV/MP3)")
    ap.add_argument("--fs", type=int, default=8000, help="sampling rate for synthetic signals")
    ap.add_argument("--bits", type=int, default=3, help="bits per DPCM sample")
    ap.add_argument("--order", type=int, default=2, help="predictor order")
    ap.add_argument("--vmax-percentile", type=float, default=100.0,
                    help="set quantizer range from this percentile of |error| (default 100 = peak)")
    ap.add_argument("--ber", type=float, default=0.0, help="BSC bit error probability")
    ap.add_argument("--ebn0", type=float, default=None, help="BPSK/AWGN Eb/N0 in dB (overrides --ber)")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--play", action="store_true",
                    help="play the original and then the DPCM output through the speakers")
    ap.add_argument("--no-sweeps", action="store_true", help="skip the parameter sweeps (faster)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    is_file = args.signal in ("wav", "mp3")
    seconds = args.seconds if args.seconds is not None else (10.0 if is_file else 1.0)

    # ---- input -------------------------------------------------------------
    source = args.signal
    if args.signal == "wav":
        if not args.wav:
            ap.error("--signal wav needs --wav <file>")
        x, fs = D.load_audio(args.wav, seconds, args.start)
        source = f"WAV file '{os.path.basename(args.wav)}'"
    elif args.signal == "mp3":
        if not args.mp3:
            ap.error("--signal mp3 needs --mp3 <file>")
        try:
            x, fs = D.load_audio(args.mp3, seconds, args.start)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            sys.exit(f"\nERROR: {exc}")
        source = f"MP3 file '{os.path.basename(args.mp3)}' (mono mix)"
    else:
        x, fs = D.make_signal(args.signal, args.fs, seconds), args.fs

    # ---- end-to-end run ------------------------------------------------------
    tx, rx_bits, x_hat, n_err = run_link(x, args.bits, args.order, args.ber, args.ebn0,
                                         vmax_percentile=args.vmax_percentile)
    p = args.order
    snr_dpcm_tx = D.snr_db(x, tx.x_rec, skip=p)
    snr_e2e = D.snr_db(x, x_hat, skip=p)
    snr_pcm = D.snr_db(x, D.pcm_roundtrip(x, args.bits))
    gain = D.prediction_gain_db(x, tx.error, skip=p)

    print("=" * 62)
    print(" DPCM END-TO-END SIMULATION")
    print("=" * 62)
    print(f" signal            : {source}")
    print(f"                     {len(x)} samples = {len(x) / fs:.2f} s, fs = {fs} Hz")
    print(f" quantizer         : {args.bits} bits/sample  ->  {2 ** args.bits} levels")
    print(f" bit rate          : {args.bits * fs} bit/s")
    print(f" predictor order   : {p}   coefficients = {np.round(tx.params.coeffs, 4)}")
    print(f" quantizer range   : +/- {tx.params.vmax:.4f}   (step = {2 * tx.params.vmax / 2 ** args.bits:.4f})")
    print(f" prediction gain   : {gain:.2f} dB")
    print("-" * 62)
    chan = (f"BPSK/AWGN @ Eb/N0 = {args.ebn0} dB" if args.ebn0 is not None
            else f"BSC with p = {args.ber}" if args.ber > 0 else "ideal (error-free)")
    print(f" channel           : {chan}")
    print(f" bits sent / errors: {len(rx_bits)} / {n_err}  (BER = {n_err / len(rx_bits):.2e})")
    print("-" * 62)
    print(f" SQNR  DPCM (transmitter local decoder) : {snr_dpcm_tx:6.2f} dB")
    print(f" SNR   DPCM end-to-end (receiver output): {snr_e2e:6.2f} dB")
    print(f" SQNR  PCM, same {args.bits} bits/sample           : {snr_pcm:6.2f} dB")
    print(f" DPCM advantage over PCM                : {snr_dpcm_tx - snr_pcm:6.2f} dB")
    print("=" * 62)

    if args.ber == 0 and args.ebn0 is None:
        assert np.allclose(tx.x_rec, x_hat), "Rx output must match Tx local decoder on a clean channel"
        print(" [check] receiver output == transmitter local reconstruction  ->  OK")

    # ---- plots & audio files -------------------------------------------------
    plot_waveforms(x, tx, x_hat, fs, p, args.out, start=len(x) // 2 if is_file else None)
    plot_histograms(x, tx, p, args.out)
    orig_path = os.path.join(args.out, "original.wav")
    out_path = os.path.join(args.out, "received.wav")
    D.save_wav(orig_path, x, fs)
    D.save_wav(out_path, x_hat, fs)

    if not args.no_sweeps:
        # long recordings: run the sweeps on a short excerpt from the middle
        xs = x
        if len(x) > SWEEP_MAX_SAMPLES:
            mid = len(x) // 2
            xs = x[mid - SWEEP_MAX_SAMPLES // 2: mid + SWEEP_MAX_SAMPLES // 2]
            print(f"\n (parameter sweeps use a {len(xs) / fs:.2f} s excerpt from the middle of the signal)")
        bits_list, s_dpcm, s_pcm = plot_snr_vs_bits(xs, p, args.out)
        plot_snr_vs_order(xs, args.bits, args.out)
        plot_snr_vs_ber(xs, args.bits, p, args.out)
        plot_snr_vs_ebn0(xs, args.bits, p, args.out)
        print("\n bits | DPCM SQNR (dB) | PCM SQNR (dB)")
        for b, a, c in zip(bits_list, s_dpcm, s_pcm):
            print(f"  {b:2d}  |    {a:7.2f}     |   {c:7.2f}")

    print(f"\n Plots and WAV files saved in: {os.path.abspath(args.out)}/")
    print("   original.wav = input excerpt    received.wav = output of the DPCM link")

    if args.play:
        print("\n Playing ORIGINAL ...")
        if play_wav(orig_path):
            print(" Playing DPCM OUTPUT ...")
            play_wav(out_path)


if __name__ == "__main__":
    main()
