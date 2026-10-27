# Nyquist explorer

A simple vibration analysis tool. Upload a recording and explore its time series, spectrum, and spectrogram, with optional FIR filtering.

![Nyquist explorer screenshot](image.png)

## Updates

* New look with a dark theme
* Updated packages and Python 3.14
* Automatic FIR filter settings

## Setup

```sh
git clone https://github.com/karanrajagopalan/Nyquist-explorer.git
cd Nyquist-explorer
```

### Run locally

Requires Python 3.14 and Poetry.

```sh
poetry env use python3.14
poetry sync
poetry run python app/Nyquist_explorer.py
```

### Run with Docker

```sh
docker build ./ -t nyquist_explorer
docker run -p 8050:8050 nyquist_explorer
```

Open [http://localhost:8050](http://localhost:8050).

## Usage

1. Upload a NetCDF (`.nc`) or CSV (`.csv`) recording.
2. Pick a signal channel and check the sampling rate.
3. Zoom the time series to analyze a selected interval, or analyze the full recording.
4. Adjust the spectrum and spectrogram settings.
5. Optionally enable a low-pass, high-pass, or band-pass FIR filter. Tap count and window are set automatically, or switch that off to set them yourself.

Use **Hide settings** to give the charts more room.

## Tests

```sh
poetry run python -m unittest discover -s tests -v
```
