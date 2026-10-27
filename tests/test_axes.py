import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from Nyquist_explorer import update_td_plot
from utils import calculate_fft


class TimeAxisTests(unittest.TestCase):
    def test_sample_times(self):
        for sample_count, fs in [(4, 4), (1, 8), (7, 50000)]:
            with self.subTest(sample_count=sample_count, fs=fs):
                samples = list(range(sample_count))
                memory = {"data": {"data_vars": {"signal": {"data": samples}}}}
                figure, filtered = update_td_plot(memory, "signal", fs, [], None)

                np.testing.assert_allclose(
                    figure.data[0].x, np.arange(sample_count) / fs
                )
                np.testing.assert_array_equal(figure.data[0].y, samples)
                self.assertEqual(filtered, {"filtered": None, "index": None})

    def test_filtered_trace_uses_corrected_sample_times(self):
        memory = {"data": {"data_vars": {"signal": {"data": [1, 2, 3, 4]}}}}
        figure, filtered = update_td_plot(
            memory, "signal", 4, [0], {"taps": [0.25, 0.5, 0.25]}
        )

        np.testing.assert_allclose(figure.data[1].x, [0, 0.25, 0.5])
        np.testing.assert_allclose(filtered["index"], [0, 0.25, 0.5])
        np.testing.assert_allclose(filtered["filtered"], [1, 2, 3])


class FrequencyAxisTests(unittest.TestCase):
    def test_bin_spacing_and_existing_output_shape(self):
        for nfft, fs in [(8, 8), (9, 18), (2048, 50000)]:
            with self.subTest(nfft=nfft, fs=fs):
                spectrum = calculate_fft(np.ones(nfft), nfft, fs)

                self.assertEqual(list(spectrum.columns), ["Amplitude", "Frequency"])
                self.assertEqual(len(spectrum), nfft // 2)
                np.testing.assert_allclose(
                    spectrum["Frequency"], np.arange(nfft // 2) * fs / nfft
                )

    def test_known_tone_peak_and_unnormalized_magnitude(self):
        fs = 64
        samples = np.sin(2 * np.pi * 8 * np.arange(64) / fs)
        spectrum = calculate_fft(samples, 64, fs)
        peak = spectrum.loc[spectrum["Amplitude"].idxmax()]

        self.assertAlmostEqual(peak["Frequency"], 8)
        self.assertAlmostEqual(peak["Amplitude"], 32)

    def test_padding_and_truncation_preserve_magnitudes(self):
        samples = np.arange(8)
        for nfft in [4, 8, 16]:
            with self.subTest(nfft=nfft):
                spectrum = calculate_fft(samples, nfft, 32)
                np.testing.assert_allclose(
                    spectrum["Amplitude"], np.abs(np.fft.fft(samples, nfft))[:nfft // 2]
                )


if __name__ == "__main__":
    unittest.main()
