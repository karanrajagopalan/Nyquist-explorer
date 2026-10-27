import base64
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from dash import no_update
from netCDF4 import Dataset
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from Nyquist_explorer import (
    app, filter_control_visibility, plot_filter, render_signal, update_channel_details,
    select_workspace, update_automatic_taps, update_channels, update_fd_plot, update_memory,
)
from utils import (
    apply_filter, automatic_fir_settings, calculate_spectrogram, calculate_spectrum, get_ftaps,
    interval_indices, parse_contents, sampling_rate,
)


def upload_contents(data):
    return "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii")


def recording(count=64, fs=64):
    return {
        "id": "test-recording", "filename": "test.csv",
        "data": {
            "data_vars": {
                "signal": {
                    "data": np.sin(2 * np.pi * 8 * np.arange(count) / fs).tolist(),
                    "attrs": {"Fs": str(fs), "units": "m/s2"},
                },
            },
            "attrs": {}, "skipped": [],
        },
    }


class UploadTests(unittest.TestCase):
    def test_csv_channels_share_netcdf_structure(self):
        data = parse_contents(upload_contents(b"vibration,name,bad\n1,a,2\n2,b,\n3,c,4\n"), "DATA.CSV")
        self.assertEqual(list(data["data_vars"]), ["vibration"])
        self.assertEqual(data["data_vars"]["vibration"]["data"], [1, 2, 3])
        self.assertEqual(len(data["skipped"]), 2)

    def test_netcdf_metadata_and_unsupported_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.nc"
            with Dataset(path, "w") as dataset:
                dataset.createDimension("time", 4)
                dataset.title = "Test recording"
                variable = dataset.createVariable("vibration", "f8", ("time",))
                variable[:] = [1, 2, 3, 4]
                variable.Fs = 128.5
                variable.units = "m/s2"
                matrix = dataset.createVariable("matrix", "f8", ("time", "time"))
                matrix[:] = np.ones((4, 4))
                missing = dataset.createVariable("missing", "f8", ("time",), fill_value=-999)
                missing[:] = [1, -999, 2, 3]
            data = parse_contents(upload_contents(path.read_bytes()), path.name)
        self.assertEqual(list(data["data_vars"]), ["vibration"])
        self.assertEqual(data["attrs"]["title"], "Test recording")
        self.assertEqual(data["data_vars"]["vibration"]["attrs"]["Fs"], "128.5")
        self.assertEqual(len(data["skipped"]), 2)

    def test_invalid_uploads_raise_explicit_errors(self):
        for contents, filename in [
            (upload_contents(b"test"), "bad.xlsx"),
            ("not-base64", "bad.nc"),
            ("data:text/plain;base64,%%%", "bad.csv"),
            (upload_contents(b"not netcdf"), "bad.nc"),
            (upload_contents(b""), "empty.csv"),
            (upload_contents(b"text\nhello\n"), "text.csv"),
            (upload_contents(b"x\ninf\nnan\n"), "nonfinite.csv"),
        ]:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                parse_contents(contents, filename)


class AnalysisTests(unittest.TestCase):
    def test_sampling_rate_allows_fractional_but_rejects_invalid_values(self):
        self.assertEqual(sampling_rate(128.5), 128.5)
        for value in [None, "", 0, -1, "nan", float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                sampling_rate(value)

    def test_normalized_tone_amplitude_with_padding(self):
        fs = 64
        samples = 3 * np.sin(2 * np.pi * 8 * np.arange(64) / fs)
        for nfft in [64, 128]:
            with self.subTest(nfft=nfft):
                frequencies, amplitude, effective = calculate_spectrum(samples, nfft, fs, "amplitude")
                index = np.flatnonzero(frequencies == 8)[0]
                self.assertAlmostEqual(amplitude[index], 3)
                self.assertEqual(effective, nfft)

    def test_amplitude_does_not_double_dc_or_nyquist(self):
        for samples, expected_index in [(np.ones(8), 0), ((-1.0) ** np.arange(8), -1)]:
            _, amplitude, _ = calculate_spectrum(samples, 8, 8, "amplitude")
            self.assertAlmostEqual(amplitude[expected_index], 1)

    def test_odd_fft_last_positive_bin_is_doubled(self):
        samples = np.cos(2 * np.pi * 4 * np.arange(9) / 9)
        frequencies, amplitude, _ = calculate_spectrum(samples, 9, 9, "amplitude")
        self.assertEqual(frequencies[-1], 4)
        self.assertAlmostEqual(amplitude[-1], 1)

    def test_psd_integrates_to_mean_square(self):
        samples = np.random.default_rng(42).normal(size=128)
        frequencies, psd, _ = calculate_spectrum(samples, 256, 64, "psd")
        self.assertAlmostEqual(np.sum(psd) * (frequencies[1] - frequencies[0]), np.mean(samples ** 2))

    def test_full_interval_never_truncates_to_fft_setting(self):
        samples = np.arange(128)
        frequencies, magnitude, effective = calculate_spectrum(samples, 8, 64, "raw")
        self.assertEqual(effective, 128)
        self.assertEqual(len(frequencies), 65)
        np.testing.assert_allclose(magnitude, np.abs(np.fft.rfft(samples)))

    def test_interval_bounds_are_inclusive_and_clipped(self):
        self.assertEqual(interval_indices(100, 10, "full", None), (0, 100))
        self.assertEqual(interval_indices(100, 10, "selected", [1, 2]), (10, 21))
        self.assertEqual(interval_indices(100, 10, "selected", [-2, 20]), (0, 100))
        self.assertEqual(interval_indices(100, 10, "selected", [0.11, 0.4]), (2, 5))
        for selection in [None, [2, 1], [1, 1.01], [20, 21], [-2, -1]]:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                interval_indices(100, 10, "selected", selection)

    def test_spectrogram_short_selection_and_zero_input(self):
        frequencies, times, db, segment = calculate_spectrogram(np.zeros(4), 64, 256, 512, 50)
        self.assertEqual(segment, 4)
        self.assertEqual(db.shape, (257, 1))
        self.assertTrue(np.all(np.isfinite(db)))
        self.assertEqual(frequencies[-1], 32)
        self.assertAlmostEqual(times[0], 2 / 64)

    def test_invalid_analysis_settings(self):
        for nfft in [None, 0, 1, 2.5]:
            with self.subTest(nfft=nfft), self.assertRaises(ValueError):
                calculate_spectrum([1, 2], nfft, 64, "raw")
        for window, nfft, overlap in [(256, 128, 50), (None, 512, 50), (2, 8, 100), (2, 8, -1)]:
            with self.subTest(window=window, nfft=nfft, overlap=overlap), self.assertRaises(ValueError):
                calculate_spectrogram([1, 2], 64, window, nfft, overlap)


class FilterTests(unittest.TestCase):
    def test_automatic_length_uses_rate_cutoffs_and_odd_kaiser_estimate(self):
        for kind, lower, upper, fs, expected_width in [
            ("lowpass", 500, None, 51200, 50),
            ("highpass", 10, None, 51200, 1),
            ("lowpass", 490, None, 1000, 1),
            ("bandpass", 100, 120, 1000, 2),
            ("bandpass", 5, 120, 1000, 0.5),
            ("bandpass", 100, 495, 1000, 0.5),
        ]:
            with self.subTest(kind=kind, lower=lower, upper=upper, fs=fs):
                count, beta, width = automatic_fir_settings(kind, lower, upper, fs)
                expected_count, expected_beta = signal.kaiserord(60, expected_width / (fs / 2))
                self.assertEqual(count, expected_count + (expected_count % 2 == 0))
                self.assertGreaterEqual(count, 3)
                self.assertEqual(count % 2, 1)
                self.assertAlmostEqual(beta, expected_beta)
                self.assertAlmostEqual(width, expected_width)
        small_rate = automatic_fir_settings("lowpass", 100, None, 1000)[0]
        large_rate = automatic_fir_settings("lowpass", 100, None, 2000)[0]
        narrow_cutoff = automatic_fir_settings("lowpass", 10, None, 1000)[0]
        self.assertGreater(large_rate, small_rate)
        self.assertGreater(narrow_cutoff, small_rate)

    def test_automatic_design_approximately_meets_attenuation_target(self):
        for kind in ["lowpass", "highpass", "bandpass"]:
            with self.subTest(kind=kind):
                count, beta, width = automatic_fir_settings(kind, 100, 200, 1000)
                taps = get_ftaps(kind, count, 100, 200, 1000, "kaiser", beta)
                frequencies, response = signal.freqz(taps, worN=65536, fs=1000)
                if kind == "lowpass":
                    stopband = frequencies >= 100 + width / 2
                elif kind == "highpass":
                    stopband = frequencies <= 100 - width / 2
                else:
                    stopband = (frequencies <= 100 - width / 2) | (frequencies >= 200 + width / 2)
                # Kaiser order is an empirical estimate, not a strict attenuation guarantee.
                self.assertLess(20 * np.log10(np.max(np.abs(response[stopband]))), -59.5)

    def test_automatic_design_rejects_invalid_cutoffs_and_rates(self):
        for args in [
            ("lowpass", None, None, 1000), ("lowpass", 0, None, 1000),
            ("highpass", 500, None, 1000), ("bandpass", 200, 100, 1000),
            ("bandpass", 100, None, 1000), ("lowpass", 100, None, 0),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                automatic_fir_settings(*args)

    def test_filter_types_and_frequency_response(self):
        for kind in ["lowpass", "highpass", "bandpass"]:
            with self.subTest(kind=kind):
                taps = get_ftaps(kind, 101, 100, 200, 1000)
                frequencies, response = signal.freqz(taps, worN=[0, 150, 500], fs=1000)
                gain = np.abs(response)
                self.assertEqual(len(taps), 101)
                if kind == "lowpass":
                    self.assertGreater(gain[0], 0.99)
                    self.assertLess(gain[-1], 0.01)
                elif kind == "highpass":
                    self.assertLess(gain[0], 0.01)
                    self.assertGreater(gain[-1], 0.99)
                else:
                    self.assertGreater(gain[1], 0.99)
                    self.assertLess(gain[0], 0.01)

    def test_kaiser_parameter_and_invalid_settings(self):
        self.assertEqual(len(get_ftaps("lowpass", 101, 100, None, 1000, "kaiser", 8.6)), 101)
        for args in [
            ("lowpass", 100, 100, None, 1000),
            ("lowpass", 101, 0, None, 1000),
            ("highpass", 101, 500, None, 1000),
            ("bandpass", 101, 200, 100, 1000),
            ("bandpass", 101, 100, None, 1000),
            ("lowpass", 101, 100, None, 1000, "kaiser", None),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                get_ftaps(*args)

    def test_delay_compensation_and_short_channel_error(self):
        filtered, times, delay = apply_filter([1, 2, 3, 4], [0.25, 0.5, 0.25], 4)
        np.testing.assert_allclose(filtered, [1, 2, 3])
        np.testing.assert_allclose(times, [0, 0.25, 0.5])
        self.assertEqual(delay, 0.25)
        with self.assertRaisesRegex(ValueError, "too short"):
            apply_filter([1, 2], np.ones(101), 64)

    def test_long_filter_matches_causal_fir_output(self):
        samples = np.random.default_rng(42).normal(size=4096)
        taps = get_ftaps("lowpass", 1025, 100, None, 1000)
        filtered, times, delay = apply_filter(samples, taps, 1000)
        expected = signal.lfilter(taps, 1.0, samples)[512:]
        np.testing.assert_allclose(filtered, expected, atol=1e-12)
        self.assertEqual(len(times), len(expected))
        self.assertEqual(delay, 0.512)


class CallbackTests(unittest.TestCase):
    def setUp(self):
        self.memory = recording()
        self.filter_data = plot_filter("lowpass", 11, 64, 12, None, "hamming", None)[2]
        self.filtered = render_signal(self.memory, "signal", 64, False, self.filter_data, 0)[1]

    def analyze(self, **overrides):
        params = dict(
            memory=self.memory, selected="signal", fs=64, nfft=8, selection=None,
            interval_mode="full", filtered_data=self.filtered, spectrum_mode="raw",
            spec_window=16, spec_nfft=32, spec_overlap=50, spec_source="raw",
        )
        params.update(overrides)
        return update_fd_plot(**params)

    def test_initial_state_and_failed_upload_are_safe(self):
        self.assertEqual(update_channels(None)[:2], ([], None))
        self.assertIsNone(update_memory(None, None)[0])
        with self.assertLogs("Nyquist_explorer", level="WARNING"):
            self.assertIsNone(update_memory(upload_contents(b"bad"), "bad.nc")[0])
        figure, filtered, feedback, summary = render_signal(None, None, None, False, None, 0)
        self.assertEqual(len(figure.data), 0)
        self.assertIn("error", filtered)
        self.assertEqual(len(self.analyze(memory=None)[0].data), 0)

    def test_script_entrypoint_uses_supported_dash_run_api(self):
        script = Path(__file__).resolve().parents[1] / "app" / "Nyquist_explorer.py"
        with patch("dash.Dash.run") as run:
            runpy.run_path(str(script), run_name="__main__")
        run.assert_called_once_with(debug=False, host="0.0.0.0", port=8050)

    def test_channel_rate_and_missing_rate_warning(self):
        self.assertEqual(update_channel_details("signal", self.memory)[0], 64)
        self.memory["data"]["data_vars"]["signal"]["attrs"] = {}
        fs, details = update_channel_details("signal", self.memory)
        self.assertEqual(fs, 50000)
        self.assertIn("No sampling-rate metadata", str(details))

    def test_channel_switch_uses_each_channels_recorded_rate(self):
        self.memory["data"]["data_vars"]["vib"] = {"data": [1, 2], "attrs": {"Fs": "51200.0"}}
        self.memory["data"]["data_vars"]["tach"] = {"data": [1, 2], "attrs": {"Fs": "1024.0"}}
        self.assertEqual(update_channel_details("vib", self.memory)[0], 51200)
        self.assertEqual(update_channel_details("tach", self.memory)[0], 1024)

    def test_invalid_filter_does_not_block_raw_signal(self):
        with self.assertLogs("Nyquist_explorer", level="WARNING"):
            invalid_filter = plot_filter("lowpass", 100, 64, 100, None, "hamming", None)[2]
        raw = render_signal(self.memory, "signal", 64, False, invalid_filter, 0)
        self.assertEqual(len(raw[0].data), 1)
        with self.assertLogs("Nyquist_explorer", level="WARNING"):
            enabled = render_signal(self.memory, "signal", 64, True, invalid_filter, 0)
        self.assertEqual(len(enabled[0].data), 0)
        self.assertIn("error", enabled[1])

    def test_full_recording_and_selected_interval(self):
        spectrum, spectrogram, feedback = self.analyze()
        self.assertEqual(len(spectrum.data[0].x), 33)
        self.assertIn("64 samples", str(feedback))
        self.assertGreater(len(spectrogram.data[0].x), 1)
        spectrum, spectrogram, feedback = self.analyze(
            interval_mode="selected", selection=[0.25, 0.5]
        )
        self.assertIn("17 samples", str(feedback))
        self.assertGreater(spectrogram.data[0].x[0], 0.25)
        self.assertLessEqual(spectrogram.data[0].x[-1], 0.5)

    def test_missing_selection_and_stale_channel_clear_old_results(self):
        with self.assertLogs("Nyquist_explorer", level="WARNING"):
            self.assertEqual(len(self.analyze(interval_mode="selected")[0].data), 0)
            self.assertEqual(len(self.analyze(fs=128)[0].data), 0)

    def test_bad_spectrogram_settings_preserve_valid_spectrum(self):
        with self.assertLogs("Nyquist_explorer", level="WARNING"):
            spectrum, spectrogram, feedback = self.analyze(spec_nfft=8)
        self.assertEqual(len(spectrum.data), 1)
        self.assertEqual(len(spectrogram.data), 0)
        self.assertIn("at least its window", str(feedback))

    def test_filter_comparison_uses_common_interval(self):
        filtered = render_signal(self.memory, "signal", 64, True, self.filter_data, 0)[1]
        spectrum, spectrogram, feedback = self.analyze(filtered_data=filtered, spec_source="filtered")
        self.assertEqual(len(spectrum.data), 2)
        self.assertEqual(len(spectrum.data[0].x), len(spectrum.data[1].x))
        self.assertIn("59 samples", str(feedback))
        self.assertIn("same available interval", str(feedback))

    def test_conditional_controls(self):
        self.assertEqual(filter_control_visibility("bandpass", "kaiser"),
                         ({"display": "block"}, {"display": "block"}))
        self.assertEqual(filter_control_visibility("lowpass", "hamming"),
                         ({"display": "none"}, {"display": "none"}))

    def test_automatic_taps_update_and_manual_values_are_preserved(self):
        count, disabled, window, window_disabled, beta, beta_disabled, feedback = (
            update_automatic_taps(True, "lowpass", 51200, 500, None)
        )
        self.assertEqual(count, 3715)
        self.assertTrue(disabled and window_disabled and beta_disabled)
        self.assertEqual(window, "kaiser")
        self.assertAlmostEqual(beta, 5.65326)
        self.assertIn("order 3,714", str(feedback))
        manual = update_automatic_taps(False, "lowpass", 51200, 10, None)
        self.assertIs(manual[0], no_update)
        self.assertIs(manual[2], no_update)
        self.assertIs(manual[4], no_update)
        self.assertEqual((manual[1], manual[3], manual[5]), (False, False, False))
        with self.assertLogs("Nyquist_explorer", level="WARNING"):
            invalid = update_automatic_taps(True, "lowpass", 1024, 600, None)
        self.assertIsNone(invalid[0])
        self.assertIn("below Nyquist", str(invalid[-1]))

    def test_workspace_sections_and_stylesheet(self):
        for view in ["recording", "plots", "settings"]:
            self.assertEqual(select_workspace(view), f"workspace workspace-{view}")
        client = app.server.test_client()
        with client.get("/assets/workspace.css") as response:
            self.assertEqual(response.status_code, 200)

    def test_header_logo_is_served_as_png_with_accessible_text(self):
        client = app.server.test_client()
        with client.get("/assets/logo.png") as response:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "image/png")
            self.assertTrue(response.get_data().startswith(b"\x89PNG\r\n\x1a\n"))
        logo = app.layout.children[0].children[0]
        self.assertEqual(logo.src, app.get_asset_url("logo.png"))
        self.assertEqual(logo.alt, "Nyquist explorer waveform logo")
        self.assertEqual((logo.width, logo.height), (418, 184))

    def test_compact_settings_labels_tooltips_and_controls(self):
        layout = app.server.test_client().get("/_dash-layout").json

        def components(node):
            if isinstance(node, dict):
                if "type" in node and "props" in node:
                    yield node
                for value in node.values():
                    yield from components(value)
            elif isinstance(node, list):
                for value in node:
                    yield from components(value)

        all_components = list(components(layout))
        ids = [node["props"]["id"] for node in all_components if "id" in node["props"]]
        self.assertEqual(len(ids), len(set(ids)))
        by_id = {node["props"]["id"]: node for node in all_components if "id" in node["props"]}
        for identifier in [
            "fs", "interval_mode", "nfft", "spectrum_mode", "spec_window", "spec_nfft",
            "spec_overlap", "spec_source", "filter_type", "fc_1", "fc_2", "tap_count",
            "filter_window", "kaiser_beta",
        ]:
            with self.subTest(identifier=identifier):
                self.assertEqual(by_id[identifier]["props"]["size"], "sm")
                info = by_id[f"{identifier}_info"]
                self.assertEqual(info["type"], "Button")
                self.assertTrue(info["props"]["aria-label"].startswith("About "))
                self.assertTrue(any(
                    node["type"] == "Label" and node["props"].get("html_for") == identifier
                    for node in all_components
                ))
        for node in all_components:
            if node["type"] == "Tooltip":
                self.assertIn(node["props"]["target"], by_id)
                self.assertEqual(node["props"]["trigger"], "hover focus")
        for identifier in ["interval_mode", "spectrum_mode", "filter_type", "filter_window", "spec_source"]:
            self.assertEqual(by_id[identifier]["type"], "Select")
        self.assertNotIn("Accordion", [node["type"] for node in all_components])
        self.assertNotIn("FormText", [node["type"] for node in all_components])
        previews = [
            node for node in all_components
            if node["type"] == "Details" and node["props"].get("className") == "filter-preview"
        ]
        self.assertEqual(len(previews), 1)
        self.assertFalse(previews[0]["props"].get("open", False))

    def test_filter_summaries_keep_explanations_in_tooltips_and_errors_visible(self):
        automatic = update_automatic_taps(True, "lowpass", 51200, 500, None)[-1]
        self.assertEqual(automatic.className, "settings-summary")
        self.assertEqual(automatic.children[0].children, "order 3,714 | transition 50 Hz")
        self.assertIn("60 dB", automatic.children[1].children[1].children)
        feedback = plot_filter("lowpass", 101, 1000, 100, None, "hamming", None)[-1]
        self.assertEqual(feedback.className, "settings-summary")
        self.assertEqual(feedback.children[0].children, "Delay: 0.05 s")
        self.assertIn("Startup transients", feedback.children[1].children[1].children)
        with self.assertLogs("Nyquist_explorer", level="WARNING"):
            error = plot_filter("lowpass", 100, 1000, 100, None, "hamming", None)[-1]
        self.assertEqual(error.color, "danger")
        self.assertIn("must be odd", error.children)

    def test_dash_endpoints_and_selection_reset(self):
        client = app.server.test_client()
        for route in ["/", "/_dash-layout", "/_dash-dependencies"]:
            self.assertEqual(client.get(route).status_code, 200)

        def selection_request(changed, relayout):
            return client.post("/_dash-update-component", json={
                "output": "selection.data",
                "outputs": {"id": "selection", "property": "data"},
                "changedPropIds": [changed],
                "inputs": [
                    {"id": "graph_1", "property": "relayoutData", "value": relayout},
                    {"id": "memory1", "property": "data", "value": self.memory},
                    {"id": "var_list_radio", "property": "value", "value": "signal"},
                    {"id": "fs", "property": "value", "value": 64},
                    {"id": "reset_selection", "property": "n_clicks", "value": 1},
                ],
            })

        zoom = {"xaxis.range[0]": 0.25, "xaxis.range[1]": 0.5}
        response = selection_request("graph_1.relayoutData", zoom)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["response"]["selection"]["data"], [0.25, 0.5])
        for changed in ["reset_selection.n_clicks", "memory1.data", "var_list_radio.value", "fs.value"]:
            response = selection_request(changed, zoom)
            self.assertIsNone(response.json["response"]["selection"]["data"])
        response = selection_request("graph_1.relayoutData", {"xaxis.autorange": True})
        self.assertIsNone(response.json["response"]["selection"]["data"])


if __name__ == "__main__":
    unittest.main()
