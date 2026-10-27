import base64
import binascii
import io
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.fft as fft
from netCDF4 import Dataset
from scipy import signal


def get_handle(file):
    return Dataset(file, 'r')


def get_variable_data(nc_handle, key):
    return np.array(nc_handle[key][:]), nc_handle[key]


def number(value, label, minimum=None, integer=False):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    if integer and not result.is_integer():
        raise ValueError(f"{label} must be a whole number.")
    return int(result) if integer else result


def sampling_rate(value):
    result = number(value, "Sampling rate")
    if result <= 0:
        raise ValueError("Sampling rate must be greater than zero.")
    return result


def samples_array(values, minimum=1):
    samples = np.asarray(values)
    if samples.ndim != 1 or samples.dtype.kind not in "iuf":
        raise ValueError("Select a one-dimensional, real numeric channel.")
    if len(samples) < minimum:
        raise ValueError(f"At least {minimum} samples are required.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("The channel contains missing or non-finite samples.")
    return samples.astype(float)


def parse_contents(contents, filename):
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".nc", ".csv"}:
        raise ValueError("Unsupported file type. Upload a NetCDF (.nc) or CSV (.csv) file.")
    try:
        _, encoded = contents.split(",", 1)
        decoded = base64.b64decode(encoded, validate=True)
    except (AttributeError, ValueError, binascii.Error) as exc:
        raise ValueError("The upload could not be decoded. Please select the file again.") from exc

    channels, skipped, attrs = {}, [], {}

    def add_channel(name, values, attributes):
        if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
            skipped.append(f"{name}: contains missing samples")
            return
        try:
            samples = samples_array(values)
        except ValueError as exc:
            skipped.append(f"{name}: {exc}")
            return
        channels[str(name)] = {"data": samples.tolist(), "attrs": attributes}

    try:
        if suffix == ".csv":
            frame = pd.read_csv(io.StringIO(decoded.decode("utf-8-sig")))
            for name in frame.columns:
                add_channel(name, frame[name].to_numpy(), {})
        else:
            with Dataset("upload", mode="r", memory=decoded) as dataset:
                attrs = {key: str(dataset.getncattr(key)) for key in dataset.ncattrs()}
                for name, variable in dataset.variables.items():
                    attributes = {
                        key: str(variable.getncattr(key)) for key in variable.ncattrs()
                    }
                    add_channel(name, variable[:], attributes)
    except (OSError, RuntimeError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Unable to read {suffix} file: {exc}") from exc
    if not channels:
        detail = "; ".join(skipped)
        raise ValueError(f"No usable numeric channels were found. {detail}".strip())
    return {"data_vars": channels, "attrs": attrs, "skipped": skipped}


def calculate_fft(array1, nfft, fs):
    fs = sampling_rate(fs)
    nfft = number(nfft, "FFT length", minimum=2, integer=True)
    samples = samples_array(array1)
    fft_abs = pd.DataFrame(np.abs(fft.fft(samples, nfft)), columns=["Amplitude"])
    fft_abs["Frequency"] = fft.fftfreq(nfft, d=1.0 / fs)
    return fft_abs.head(nfft // 2)


def calculate_spectrum(values, nfft, fs, mode):
    samples = samples_array(values, minimum=2)
    fs = sampling_rate(fs)
    nfft = max(number(nfft, "FFT length", minimum=2, integer=True), len(samples))
    frequencies = fft.rfftfreq(nfft, d=1.0 / fs)
    if mode == "psd":
        frequencies, magnitudes = signal.periodogram(
            samples, fs=fs, window="boxcar", detrend=False, nfft=nfft, scaling="density"
        )
    elif mode in {"raw", "amplitude"}:
        magnitudes = np.abs(fft.rfft(samples, nfft))
        if mode == "amplitude":
            magnitudes /= len(samples)
            magnitudes[1:-1 if nfft % 2 == 0 else None] *= 2
    else:
        raise ValueError("Select a spectrum mode.")
    return frequencies, magnitudes, nfft


def interval_indices(sample_count, fs, mode, selection):
    fs = sampling_rate(fs)
    if mode == "full":
        return 0, sample_count
    if mode != "selected":
        raise ValueError("Select an analysis interval mode.")
    if not selection:
        raise ValueError("Zoom into the time-series plot to select an analysis interval.")
    left = number(selection[0], "Interval start")
    right = number(selection[1], "Interval end")
    if right <= left:
        raise ValueError("The selected interval must have a positive duration.")
    # Plotly's visible bounds include both endpoints.
    start = max(0, int(np.ceil(left * fs - 1e-9)))
    stop = min(sample_count, int(np.floor(right * fs + 1e-9)) + 1)
    if stop - start < 2:
        raise ValueError("The selected interval must contain at least two available samples.")
    return start, stop


def filter_cutoffs(filter_type, fc_1, fc_2, fs):
    fs = sampling_rate(fs)
    if filter_type not in {"lowpass", "highpass", "bandpass"}:
        raise ValueError("Select a filter type.")
    cutoff = [number(fc_1, "Cutoff frequency")]
    if filter_type == "bandpass":
        cutoff.append(number(fc_2, "Upper cutoff frequency"))
    if any(value <= 0 or value >= fs / 2 for value in cutoff):
        raise ValueError(f"Cutoff frequencies must be greater than 0 and below Nyquist ({fs / 2:g} Hz).")
    if len(cutoff) == 2 and cutoff[0] >= cutoff[1]:
        raise ValueError("The lower cutoff must be below the upper cutoff.")
    return cutoff


def automatic_fir_settings(filter_type, fc_1, fc_2, fs):
    fs = sampling_rate(fs)
    cutoff = filter_cutoffs(filter_type, fc_1, fc_2, fs)
    spans = [cutoff[0], fs / 2 - cutoff[-1]]
    if len(cutoff) == 2:
        spans.append(cutoff[1] - cutoff[0])
    transition_hz = 0.1 * min(spans)
    normalized_width = transition_hz / (fs / 2)
    if normalized_width <= 0 or not np.isfinite(8 / normalized_width):
        raise ValueError("Cutoffs are too close to an edge to estimate a finite FIR length.")
    tap_count, beta = signal.kaiserord(60, normalized_width)
    tap_count = max(3, tap_count)
    if tap_count % 2 == 0:
        tap_count += 1
    return tap_count, beta, transition_hz


def get_ftaps(filter_type, tap_count, fc_1, fc_2, fs, window="hamming", beta=8.6):
    fs = sampling_rate(fs)
    tap_count = number(tap_count, "FIR tap count", minimum=3, integer=True)
    if tap_count % 2 == 0:
        raise ValueError("FIR tap count must be odd for integer-sample delay compensation.")
    cutoff = filter_cutoffs(filter_type, fc_1, fc_2, fs)
    if window == "kaiser":
        window = ("kaiser", number(beta, "Kaiser beta", minimum=0))
    elif window not in {
        "boxcar", "hann", "hamming", "blackman", "blackmanharris", "flattop", "bartlett"
    }:
        raise ValueError("Select a supported FIR window.")
    return signal.firwin(
        tap_count, cutoff, window=window, pass_zero=filter_type, fs=fs
    )


def apply_filter(values, taps, fs):
    samples = samples_array(values)
    fs = sampling_rate(fs)
    taps = samples_array(taps, minimum=3)
    delay = (len(taps) - 1) // 2
    if len(taps) % 2 == 0:
        raise ValueError("FIR tap count must be odd.")
    if len(samples) <= delay + 1:
        raise ValueError("The channel is too short for this FIR filter. Reduce the tap count.")
    filtered = signal.convolve(samples, taps, mode="full", method="auto")[delay:len(samples)]
    return filtered, np.arange(len(filtered)) / fs, delay / fs


def calculate_spectrogram(values, fs, window_samples, nfft, overlap):
    samples = samples_array(values, minimum=2)
    fs = sampling_rate(fs)
    requested = number(window_samples, "Spectrogram window length", minimum=2, integer=True)
    nfft = number(nfft, "Spectrogram FFT length", minimum=2, integer=True)
    overlap = number(overlap, "Spectrogram overlap", minimum=0)
    if overlap >= 100:
        raise ValueError("Spectrogram overlap must be below 100%.")
    if nfft < requested:
        raise ValueError("Spectrogram FFT length must be at least its window length.")
    segment = min(requested, len(samples))
    frequencies, times, power = signal.spectrogram(
        samples, fs=fs, window="hamming", nperseg=segment,
        noverlap=int(segment * overlap / 100), nfft=nfft,
        detrend=False, scaling="density"
    )
    return frequencies, times, 10 * np.log10(np.maximum(power, np.finfo(float).tiny)), segment
