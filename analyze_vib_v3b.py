#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from typing import Optional, Tuple, Dict, List
import math
import numpy as np
from numpy.typing import NDArray
import scipy.signal as sig
import matplotlib.pyplot as plt


# ---- UFF reader (pyuff) ----
def _try_import_pyuff():
    try:
        import pyuff
        return pyuff
    except Exception:
        return None


# ---------- Helpers ----------
def db20(v: NDArray[np.float64], floor: float = -300.0) -> NDArray[np.float64]:
    """Convert to 20*log10 with a lower floor to avoid -inf."""
    v = np.maximum(np.asarray(v, dtype=np.float64), 1e-300)
    out = 20.0 * np.log10(v)
    if floor is not None:
        out = np.maximum(out, floor)
    return out


def next_pow2(n: int) -> int:
    """Next power of two >= n."""
    return 1 << (n - 1).bit_length()


def slice_time(
    x: NDArray[np.float64],
    fs: float,
    t0: Optional[float],
    t1: Optional[float],
) -> Tuple[NDArray[np.float64], Tuple[float, float]]:
    """Select time window [t0, t1]."""
    n = len(x)
    if t0 is None and t1 is None:
        return x, (0.0, (n - 1) / fs)

    i0 = 0 if t0 is None else max(0, int(round(t0 * fs)))
    i1 = n if t1 is None else min(n, int(round(t1 * fs)))
    i1 = max(i1, i0 + 1)
    return x[i0:i1], (i0 / fs, (i1 - 1) / fs)


def make_formatter(use_decimal_comma: bool):
    """
    Number formatter for log/terminal output.
    Keeps English text, but replaces '.' with ',' for floats (German notation).
    """
    def fmt(val, spec: str = ".6f") -> str:
        try:
            v = float(val)
        except Exception:
            return str(val)
        s = format(v, spec)
        if use_decimal_comma:
            s = s.replace(".", ",")
        return s
    return fmt


# ---------- UFF/UNV/WAV readers ----------
def read_uff_timehistory(path: str) -> Tuple[float, NDArray[np.float64], str]:
    """
    Read UFF/UNV time history (Dataset 58/58b, ASCII or binary).
    Determine fs from:
      - 'abscissa_inc' (Δt) or
      - 'dx' / 'dt' or
      - differences in 'x'
    Returns (fs, signal, y_label).
    """
    pyuff = _try_import_pyuff()
    if pyuff is None:
        raise RuntimeError("pyuff is not installed. Please run `pip install pyuff`.")

    uff = pyuff.UFF(path)
    fs: Optional[float] = None
    signal: Optional[NDArray[np.float64]] = None
    y_label = "Signal"

    for d in uff.read_sets():
        # pyuff uses type==58 also for 58b
        if int(d.get("type", -1)) == 58:
            y = np.asarray(d.get("data", None), dtype=np.float64).squeeze()
            if y is None or y.size == 0:
                continue

            # Determine Δt
            dt: Optional[float] = None
            for key in ("abscissa_inc", "dx", "dt"):
                if key in d and d[key] not in (None, 0, 0.0):
                    dt = float(d[key])
                    break

            if (dt is None or dt <= 0) and "x" in d and d["x"] is not None and len(d["x"]) > 1:
                xvec = np.asarray(d["x"], dtype=np.float64).squeeze()
                dxs = np.diff(xvec)
                dt = float(np.median(dxs))

            if dt is None or dt <= 0:
                raise RuntimeError("Could not determine sampling interval (dt) from UNV dataset 58.")

            fs = 1.0 / dt

            lab = (d.get("ordinate_label") or d.get("y_label") or "Signal")
            unit = (d.get("ordinate_units") or d.get("y_units") or "").strip()
            lab = (str(lab).strip() if lab is not None else "Signal")
            y_label = lab + (f" [{unit}]" if unit else "")
            signal = y
            break

    if fs is None or signal is None:
        raise RuntimeError("No valid time-history (type 58) found in UFF/UNV file.")
    return float(fs), signal, y_label


def read_wav(path: str) -> Tuple[float, NDArray[np.float64], str]:
    """Simple WAV reader (mono)."""
    from scipy.io import wavfile
    fs, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype.kind in "iu":
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    return float(fs), data, "Amplitude [a.u.]"


def read_signal(path: str) -> Tuple[float, NDArray[np.float64], str]:
    """Read signal and sampling frequency from file."""
    low = path.lower()
    if low.endswith(".unv") or low.endswith(".uff") or low.endswith(".pvd"):
        return read_uff_timehistory(path)
    if low.endswith(".wav"):
        return read_wav(path)
    raise RuntimeError(f"Unsupported file format: {path}")


# ---------- FFT & peak stats ----------
def compute_fft(
    x: NDArray[np.float64],
    fs: float,
    fmin: float,
    fmax: float,
    window: str = "flattop",
    detrend: bool = True,
    nfft: Optional[int] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """
    Compute single-sided amplitude spectrum in [fmin, fmax].
    
    Amplitude: linear (Sinus, der exakt auf einem FFT-Bin liegt, erscheint mit seiner realen Amplitude)
    Korrektur: Window coherent gain (sum(window)) und einseitiges Spektrum
    (DC/Nyquist werden nicht verdoppelt, alle anderen Bins schon).
    """
    x = np.asarray(x, dtype=np.float64)
    if detrend:
        x = sig.detrend(x, type="linear")

    win = sig.get_window(window, x.size, fftbins=True)
    xw = x * win

    if nfft is None:
        nfft = next_pow2(len(xw))


    """
    Amplitudenscaling für Flattop window
    """
    X = np.fft.rfft(xw, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # Amplitude scaling (single-sided, linear) mit "coherent gain"
    # G = sum(win)/N sorgt dafür, dass ein Sinus, der exakt auf einem FFT-Bin liegt,
    # wieder mit seiner echten Amplitude herauskommt (unabhängig vom Fenster).
    G = np.sum(win) / nfft  # coherent gain des Fensters
    if G == 0.0:
        raise RuntimeError("Window coherent gain is zero, cannot scale spectrum.")

    mag = np.abs(X) * (2.0 / (nfft * G))

    """
    
    """
    
    lo = np.searchsorted(freqs, fmin, side="left")
    hi = np.searchsorted(freqs, fmax, side="right")
    freqs = freqs[lo:hi]
    mag = mag[lo:hi]

    rbw = fs / nfft
    return freqs, mag, rbw


def peak_stats_fft(
    x: NDArray[np.float64],
    fs: float,
    fmin: float,
    fmax: float,
    window: str = "flattop",
    nfft: Optional[int] = None,
    n_segments: int = 8,
    overlap: float = 0.5,
) -> Dict[str, float]:
    """
    Split time window into several segments and determine the peak per segment.
    Für jedes Segment wird die FFT im Bereich [fmin, fmax] berechnet und die max. Amplitude gewählt.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n_segments < 1:
        n_segments = 1

    seg_len = max(1, n // n_segments)
    step = max(1, int(seg_len * (1.0 - overlap)))

    freqs_all = []
    amps_all = []
    rbw = np.nan

    for start in range(0, n - seg_len + 1, step):
        seg = x[start:start + seg_len]
        f, A, rbw = compute_fft(seg, fs, fmin, fmax, window=window, detrend=True, nfft=nfft)
        if A.size == 0:
            continue
        idx = int(np.argmax(A))
        freqs_all.append(f[idx])
        amps_all.append(A[idx])

    if len(freqs_all) == 0:
        return {
            "f_mean": np.nan,
            "f_std": np.nan,
            "A_mean": np.nan,
            "A_std": np.nan,
            "n_segments": 0,
            "rbw": rbw,
        }

    freqs_all_arr = np.array(freqs_all, dtype=np.float64)
    amps_all_arr = np.array(amps_all, dtype=np.float64)

    f_mean = float(np.mean(freqs_all_arr))
    f_std = float(np.std(freqs_all_arr, ddof=1)) if len(freqs_all_arr) > 1 else 0.0
    A_mean = float(np.mean(amps_all_arr))
    A_std = float(np.std(amps_all_arr, ddof=1)) if len(amps_all_arr) > 1 else 0.0

    return {
        "f_mean": f_mean,
        "f_std": f_std,
        "A_mean": A_mean,
        "A_std": A_std,
        "n_segments": len(freqs_all_arr),
        "rbw": rbw,
    }


# ---------- Plot & CSV export ----------
def plot_spectrum(
    f: NDArray[np.float64],
    y: NDArray[np.float64],
    peak_f: float,
    peak_amp: float,
    rbw: float,
    title: str,
    out_png: Optional[str],
    color_spectrum: str = "C0",
    color_peak: str = "red",
):
    """
    Plot magnitude spectrum in dB, mark peak, save as PNG.
    Spectrum is drawn in front, peak line in the background.
    """
    plt.figure(figsize=(9, 5))

    # Plot peak line first (background)
    plt.axvline(
        peak_f,
        linestyle="--",
        alpha=0.9,
        color=color_peak,
        label=f"Peak: {peak_f:.2f} Hz",
        zorder=1,
    )

    # Spectrum in foreground
    plt.plot(
        f,
        db20(y),
        label="Spectrum",
        color=color_spectrum,
        zorder=10,
    )

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    txt = f"Peak A = {20*np.log10(max(peak_amp, 1e-300)):.2f} dB\nRBW ≈ {rbw:.3f} Hz"
    ax = plt.gca()
    ax.text(
        0.98,
        0.98,
        txt,
        ha="right",
        va="top",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save PNG only if filename given
    if out_png is not None:
        plt.savefig(out_png, dpi=150)

    # Always show plot interactively
    plt.show()


def save_csv(
    path: str,
    f: NDArray[np.float64],
    y: NDArray[np.float64],
    use_decimal_comma: bool = False,
):
    """
    Save spectrum as CSV:
      columns: f_Hz, Amplitude_linear, Amplitude_dB
    If use_decimal_comma=True:
      - use ';' as separator
      - use ',' as decimal separator
      - header text remains English
    """
    amp_db = db20(y)
    data = np.column_stack([f, y, amp_db])
    header = "f_Hz,Amplitude_linear,Amplitude_dB"

    if not use_decimal_comma:
        # Standard: comma as separator, dot as decimal
        np.savetxt(
            path,
            data,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.9e",
        )
    else:
        # Decimal comma + semicolon separator (German Excel style)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header.replace(",", ";") + "\n")
            for row in data:
                str_vals = [format(float(val), ".9e") for val in row]
                str_vals = [s.replace(".", ",") for s in str_vals]
                fh.write(";".join(str_vals) + "\n")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Reduced FFT analysis of LDV data (UNV/UFF/WAV): peak + mean/std."
    )
    ap.add_argument(
        "--decimal-comma-log",
        action="store_true",
        help="Use comma as decimal separator in console log output.",
    )
    ap.add_argument("input", help="Input file (.unv/.uff/.pvd or .wav)")
    ap.add_argument("--t0", type=float, default=None, help="Start time [s] of window")
    ap.add_argument("--t1", type=float, default=None, help="End time [s] of window")
    ap.add_argument("--fmin", type=float, required=True, help="Minimum frequency [Hz]")
    ap.add_argument("--fmax", type=float, required=True, help="Maximum frequency [Hz]")
    ap.add_argument(
        "--ftarget",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional: eine oder mehrere Zielfrequenzen [Hz] für "
            "Amplitude- und Bin-Check (z.B. --ftarget 500 1000 1500 2000)."
        ),
    )
    ap.add_argument(
        "--bin-tol",
        type=float,
        default=1e-6,
        help="Toleranz in Bin-Einheiten für 'on-bin' (Default: 1e-6).",
    )
    ap.add_argument(
        "--window",
        default="flattop",
        help="FFT window (scipy.signal.get_window, default: flattop)",
    )
    ap.add_argument(
        "--nfft",
        type=int,
        default=None,
        help="NFFT (default: next power of two of window length)",
    )
    ap.add_argument(
        "--segments",
        type=int,
        default=8,
        help="Number of segments for peak mean/std (default: 8)",
    )
    ap.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Segment overlap fraction [0..1], default: 0.5",
    )
    ap.add_argument(
        "--csv",
        action="store_true",
        help="Write CSV file with spectrum (default: off)",
    )
    ap.add_argument(
        "--png",
        action="store_true",
        help="Write PNG file with spectrum (default: off). Plot is always shown.",
    )
    ap.add_argument(
        "--decimal-comma",
        action="store_true",
        help="Use ',' as decimal separator and ';' as field separator in CSV and log "
             "(only applies to numeric output; text remains English).",
    )
    ap.add_argument(
        "--suffix",
        default="_spectrum",
        help="Suffix for output files (default: _spectrum). "
             "Used for CSV, PNG and log file, e.g. <input><suffix>.csv/.png/.txt.",
    )
    ap.add_argument(
        "--color-spectrum",
        default="C0",
        help="Matplotlib color for spectrum line (default: C0).",
    )
    ap.add_argument(
        "--color-peak",
        default="red",
        help="Matplotlib color for peak marker (default: red).",
    )

    args = ap.parse_args()

    # formatter for numeric output (terminal + log)
    fmt = make_formatter(args.decimal_comma)

    # simple logger: collect all printed lines for the log file
    log_lines: List[str] = []

    def log(msg: str = ""):
        print(msg)
        log_lines.append(msg)

    # Read signal and fs from file
    try:
        fs, x, y_label = read_signal(args.input)
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)

    base, _ = os.path.splitext(args.input)
    suffix = args.suffix if args.suffix is not None else ""
    log_path = base + suffix + ".txt"
    csv_path = base + suffix + ".csv" if args.csv else None
    png_path = base + suffix + ".png" if args.png else None

    # Check Nyquist
    if args.fmax > fs * 0.499:
        log(
            f"[Info] fmax ({fmt(args.fmax, '.1f')} Hz) > Nyquist ({fmt(fs/2, '.1f')} Hz). "
            f"Clamping to {fmt(fs/2 - 1.0, '.1f')} Hz."
        )
        args.fmax = max(10.0, fs / 2 - 1.0)

    # Time window
    xw, (t_start, t_end) = slice_time(x, fs, args.t0, args.t1)

    # FFT over full window
    f, Y, rbw = compute_fft(
        xw,
        fs,
        args.fmin,
        args.fmax,
        window=args.window,
        detrend=True,
        nfft=args.nfft,
    )

    if Y.size == 0:
        log("[Warning] Spectrum is empty in the selected frequency range.")
        # write log before exit
        try:
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(log_lines) + "\n")
            print(f"\nLog written to: {log_path}")
        except Exception as e:
            print(f"[Error] Could not write log file: {e}", file=sys.stderr)
        sys.exit(0)
        
    # Globaler Peak im ausgewerteten Spektrum
    peak_idx = int(np.argmax(Y))
    peak_f = float(f[peak_idx])
    peak_amp = float(Y[peak_idx])
    peak_db = 20 * math.log10(max(peak_amp, 1e-300))
    peak_rms = peak_amp / math.sqrt(2.0)
    peak_rms_db = 20 * math.log10(max(peak_rms, 1e-300))
    
    # Amplitude/Bin-Check für eine oder mehrere Ziel-Frequenzen
    if args.ftarget is not None:
        # Effektives NFFT aus RBW zurückrechnen
        nfft_eff = int(round(fs / rbw))

        log("")
        log("=== Bin alignment check(s) ===")
        log(f"FFT RBW (fs/N):         {fmt(rbw, '.6f')} Hz")
        log(f"Effective NFFT:         {nfft_eff:d}")

        # maximal 4 Targets explizit ausgeben
        for i, ftarget in enumerate(args.ftarget[:4], start=1):
            # nächstgelegener FFT-Bin im ausgewerteten Frequenzbereich
            if ftarget < f[0] or ftarget > f[-1]:
                log("")
                log(f"[Target {i}] ftarget={fmt(ftarget, '.6f')} Hz "
                    "liegt außerhalb des ausgewerteten Bereichs; "
                    "es wird der nächstgelegene Rand-Bin verwendet.")
            target_idx = int(np.argmin(np.abs(f - ftarget)))
            target_f = float(f[target_idx])
            target_amp = float(Y[target_idx])
            target_db = 20 * math.log10(max(target_amp, 1e-300))

            # theoretischer Binindex (über fs, nfft_eff)
            k_target_float = ftarget * nfft_eff / fs
            k_target_round = int(round(k_target_float))

            # Liegt die Zielfrequenz praktisch exakt auf einem Bin?
            bin_aligned = abs(k_target_float - k_target_round) <= args.bin_tol

            # Mittelpunkt-Frequenz dieses Bins
            f_bin_center = k_target_round * fs / nfft_eff

            # Binindex des globalen Peaks (aus Peak-Frequenz zurückgerechnet)
            k_peak = int(round(peak_f * nfft_eff / fs))
            same_bin = (k_peak == k_target_round)

            df_target_bin = ftarget - f_bin_center
            df_peak_target = peak_f - ftarget
            df_peak_bin = peak_f - f_bin_center

            log("")
            log(f"[Target {i}]")
            log(f"  Requested f_target:       {fmt(ftarget, '.6f')} Hz")
            log(f"  Nearest FFT-bin freq:     {fmt(target_f, '.6f')} Hz")
            log(f"  Amplitude (nearest bin):  {fmt(target_amp, '.3e')} "
                f"({fmt(target_db, '.2f')} dB)")
            log(f"  Bin center freq f_bin:    {fmt(f_bin_center, '.6f')} Hz")
            log(f"  Global peak f_peak:       {fmt(peak_f, '.6f')} Hz")
            log(f"  Δf(target, bin):          {fmt(df_target_bin, '.6f')} Hz")
            log(f"  Δf(peak, target):         {fmt(df_peak_target, '.6f')} Hz")
            log(f"  Δf(peak, bin):            {fmt(df_peak_bin, '.6f')} Hz")
            log(
                "  Target exactly on bin?    "
                f"{'YES' if bin_aligned else 'NO'} "
                f"(tol={args.bin_tol:g} bins)"
            )
            log(f"  Peak uses same bin?       {'YES' if same_bin else 'NO'}")
            log(
                "  |f_peak - f_target| <= RBW/2?  "
                f"{'YES' if abs(df_peak_target) <= rbw / 2.0 else 'NO'}"
            )



    # Peak statistics over segments
    stats = peak_stats_fft(
        xw,
        fs,
        args.fmin,
        args.fmax,
        window=args.window,
        nfft=args.nfft,
        n_segments=args.segments,
        overlap=args.overlap,
    )

    log("\n=== FFT Analysis ===")
    log(f"File:          {os.path.basename(args.input)}")
    log(f"Signal label:  {y_label}")
    log(f"fs:            {fmt(fs, '.3f')} Hz")
    log(
        f"Time window:   {fmt(t_start, '.6f')} s – {fmt(t_end, '.6f')} s "
        f"(duration {fmt(len(xw) / fs, '.6f')} s)"
    )
    log(
        f"Freq. range:   {fmt(args.fmin, '.2f')} Hz – {fmt(args.fmax, '.2f')} Hz"
    )
    log(f"RBW (≈):       {fmt(rbw, '.6f')} Hz")

    log("\n--- Peak (full spectrum) ---")
    log(f"Peak frequency:        {fmt(peak_f, '.2f')} Hz")
    log(f"Peak amplitude (lin.): {fmt(peak_amp, '.6e')}")
    log(f"Peak amplitude (dB):   {fmt(peak_db, '.2f')} dB")
    log(f"Peak RMS (lin.):       {fmt(peak_rms, '.6e')}")
    log(f"Peak RMS (dB):         {fmt(peak_rms_db, '.2f')} dB")

    log("\n--- Peak statistics over segments ---")
    log(f"Segments used:         {stats['n_segments']}")
    log(
        f"Peak frequency:        {fmt(stats['f_mean'], '.2f')} Hz ± "
        f"{fmt(stats['f_std'], '.2f')} Hz (std)"
    )
    log(
        f"Peak amplitude (lin.): {fmt(stats['A_mean'], '.3e')} ± "
        f"{fmt(stats['A_std'], '.3e')} (std)"
    )
    log(f"RBW (segment FFT):     {fmt(stats['rbw'], '.6f')} Hz")

    # --- CSV output ---
    if csv_path is not None:
        save_csv(
            csv_path,
            f,
            Y,
            use_decimal_comma=args.decimal_comma,
        )
        log(f"\nCSV saved to: {csv_path}")
        if args.decimal_comma:
            log("  -> using ';' as separator and ',' as decimal separator.")

    # --- PNG output ---
    # Plot is always shown; PNG saved only if args.png is set.
    plot_spectrum(
        f,
        Y,
        peak_f=peak_f,
        peak_amp=peak_amp,
        rbw=rbw,
        title=os.path.basename(args.input),
        out_png=png_path,
        color_spectrum=args.color_spectrum,
        color_peak=args.color_peak,
    )
    if png_path is not None:
        log(f"PNG saved to: {png_path}")

    # --- Write log file ---
    try:
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(log_lines) + "\n")
        print(f"\nLog written to: {log_path}")
    except Exception as e:
        print(f"[Error] Could not write log file: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
