import logging
from uuid import uuid4

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, ctx, dcc, html
from scipy import signal
from scipy.fft import next_fast_len

from left_panel import compact_feedback, file_details, files_location, filter_panel, right_panel
from utils import (
    apply_filter, automatic_fir_settings, calculate_spectrogram, calculate_spectrum, get_ftaps,
    interval_indices, number, parse_contents, samples_array, sampling_rate,
)

logger = logging.getLogger(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Nyquist explorer"
server = app.server


def empty_figure(message):
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=35, b=40),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return figure


def style_figure(figure, title, x_title, y_title):
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=title, xaxis_title=x_title, yaxis_title=y_title,
        margin=dict(l=50, r=20, t=45, b=45),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return figure


def alert(message, color="danger"):
    return dbc.Alert(message, color=color, className="py-2 small")


def channel_data(memory, selected):
    if not memory:
        raise ValueError("Upload a NetCDF or CSV recording to begin.")
    channels = memory["data"]["data_vars"]
    if selected not in channels:
        raise ValueError("Select a signal channel.")
    return channels[selected]


def channel_units(channel):
    attrs = channel.get("attrs", {})
    return attrs.get("units", attrs.get("unit", "a.u."))


app.layout = html.Div(
    [
        html.Header([
            html.Img(src=app.get_asset_url("logo.png"), alt="Nyquist explorer waveform logo",
                     className="app-logo", width=418, height=184),
            html.Div([
                html.H3("ℕ𝕪𝕢𝕦𝕚𝕤𝕥 𝕖𝕩𝕡𝕝𝕠𝕣𝕖𝕣", **{"aria-label": "Nyquist explorer"}),
            ], className="app-brand"),
        ], className="app-header"),
        dcc.RadioItems(
            id="workspace_view",
            options=[
                {"label": "Recording", "value": "recording"},
                {"label": "Plots", "value": "plots"},
                {"label": "Settings", "value": "settings"},
            ],
            value="plots", className="workspace-navigation", inline=True,
        ),
        dcc.Store(id="memory1"),
        dcc.Store(id="filter_mem"),
        dcc.Store(id="filt_x_mem"),
        dcc.Store(id="selection"),
        html.Main(
            [
                html.Section(
                    [html.H5("Recording"), files_location, file_details],
                    className="recording-panel workspace-panel card",
                    **{"aria-label": "Recording"},
                ),
                html.Section(
                    [
                        dbc.Button(
                            "Hide settings ▸", id="settings_toggle", color="secondary", outline=True,
                            size="sm", className="settings-toggle align-self-end", n_clicks=0,
                        ),
                        html.Div([
                            html.Div(id="signal_feedback", role="status"),
                            html.Div(id="channel_info", className="border rounded p-2 small", role="status"),
                            html.Div(id="analysis_feedback", role="status"),
                        ], className="analysis-status"),
                        html.Div(
                            dcc.Loading(
                                dbc.Tabs([
                                    dbc.Tab(html.Div([
                                        dcc.Graph(id="graph_1", responsive=True, figure=empty_figure("Upload a recording"),
                                                  className="analysis-graph"),
                                        dcc.Graph(id="graph_2", responsive=True, figure=empty_figure("Spectrum appears here"),
                                                  className="analysis-graph"),
                                    ], className="time-spectrum-plots"), label="Time series & spectrum",
                                        tab_id="time-spectrum"),
                                    dbc.Tab(
                                        dcc.Graph(id="graph_spec", responsive=True, figure=empty_figure("Spectrogram appears here"),
                                                  className="analysis-graph"),
                                        label="Spectrogram",
                                        tab_id="spectrogram",
                                    ),
                                ], className="plot-tabs", active_tab="time-spectrum"),
                                type="circle", parent_className="plots-loading",
                            ),
                            className="plots-card card",
                        ),
                    ],
                    className="plots-panel workspace-panel",
                    **{"aria-label": "Analysis plots"},
                ),
                html.Section(
                    [right_panel, filter_panel],
                    className="settings-panel workspace-panel",
                    **{"aria-label": "Settings"},
                ),
            ],
            id="workspace", className="workspace workspace-plots",
        ),
    ],
    className="app-shell",
)


@app.callback(
    Output("workspace", "className"), Output("settings_toggle", "children"),
    Input("workspace_view", "value"), Input("settings_toggle", "n_clicks"),
)
def select_workspace(view, clicks):
    collapsed = bool(clicks and clicks % 2)
    classes = f"workspace workspace-{view}" + (" settings-collapsed" if collapsed else "")
    return classes, "◂ Show settings" if collapsed else "Hide settings ▸"


@app.callback(
    Output("memory1", "data"), Output("upload_feedback", "children"),
    Input("upload", "contents"), dash.State("upload", "filename"),
)
def update_memory(contents, filename):
    if not contents:
        return None, html.Small("Supported: .nc and numeric-column .csv files.")
    try:
        data = parse_contents(contents, filename)
    except ValueError as exc:
        logger.warning("Upload rejected: %s", exc)
        return None, alert(str(exc))
    feedback = [alert(f"Loaded {filename}", "success")]
    if data["skipped"]:
        feedback.append(alert("Skipped channels: " + "; ".join(data["skipped"]), "warning"))
    return {"id": uuid4().hex, "filename": filename, "data": data}, feedback


@app.callback(
    Output("var_list_radio", "options"), Output("var_list_radio", "value"),
    Output("metadata", "children"), Input("memory1", "data"),
)
def update_channels(memory):
    if not memory:
        return [], None, "No file loaded."
    channels = list(memory["data"]["data_vars"])
    attrs = memory["data"]["attrs"]
    metadata = html.Dl([
        html.Div([html.Dt(key), html.Dd(value, style={"overflowWrap": "anywhere"})])
        for key, value in attrs.items()
    ]) if attrs else "No global metadata."
    return [{"label": name, "value": name} for name in channels], channels[0], metadata


@app.callback(
    Output("fs", "value"), Output("variable_content", "children"),
    Input("var_list_radio", "value"), Input("memory1", "data"),
)
def update_channel_details(selected, memory):
    if not memory or selected not in memory["data"]["data_vars"]:
        return 50000, "Choose a channel after uploading."
    channel = channel_data(memory, selected)
    attrs = channel.get("attrs", {})
    details = [html.Div(f"{len(channel['data']):,} samples; units: {channel_units(channel)}")]
    details.extend(html.Div(f"{key}: {value}") for key, value in attrs.items())
    if "unitscale" in attrs:
        details.append(html.Div("Values are shown as stored; unitscale is not applied."))
    recorded_rate = attrs.get("Fs", attrs.get("fs", attrs.get("sampling_rate")))
    if recorded_rate is None:
        details.append(alert("No sampling-rate metadata. Verify the default rate before analysis.", "warning"))
        return 50000, details
    try:
        fs = sampling_rate(recorded_rate)
    except ValueError as exc:
        logger.warning("Invalid channel sampling-rate metadata: %s", exc)
        details.append(alert("Invalid sampling-rate metadata. Enter the correct rate.", "warning"))
        return None, details
    return fs, details


@app.callback(
    Output("upper_cutoff", "style"), Output("kaiser_settings", "style"),
    Input("filter_type", "value"), Input("filter_window", "value"),
)
def filter_control_visibility(filter_type, window):
    return (
        {"display": "block" if filter_type == "bandpass" else "none"},
        {"display": "block" if window == "kaiser" else "none"},
    )


@app.callback(
    Output("tap_count", "value"), Output("tap_count", "disabled"),
    Output("filter_window", "value"), Output("filter_window", "disabled"),
    Output("kaiser_beta", "value"), Output("kaiser_beta", "disabled"),
    Output("auto_taps_feedback", "children"),
    Input("auto_taps", "value"), Input("filter_type", "value"), Input("fs", "value"),
    Input("fc_1", "value"), Input("fc_2", "value"),
)
def update_automatic_taps(enabled, filter_type, fs, fc_1, fc_2):
    if not enabled:
        return (
            dash.no_update, False, dash.no_update, False, dash.no_update, False,
            html.Small("Manual tap count", className="settings-summary"),
        )
    try:
        tap_count, beta, transition_hz = automatic_fir_settings(filter_type, fc_1, fc_2, fs)
    except ValueError as exc:
        logger.warning("Cannot estimate automatic FIR length: %s", exc)
        return None, True, "kaiser", True, None, True, alert(str(exc))
    message = (
        f"{tap_count:,} taps (order {tap_count - 1:,}); transition width {transition_hz:g} Hz; "
        "60 dB attenuation target, Kaiser estimate. "
        "Width is 10% of the smallest cutoff-to-edge or band-pass span. "
        "Narrow transitions require more taps and a longer recording."
    )
    summary = f"order {tap_count - 1:,} | transition {transition_hz:g} Hz"
    feedback = compact_feedback("auto_design", summary, message)
    return tap_count, True, "kaiser", True, beta, True, feedback


@app.callback(
    Output("graph_3", "figure"), Output("graph_4", "figure"),
    Output("filter_mem", "data"), Output("filter_feedback", "children"),
    Input("filter_type", "value"), Input("tap_count", "value"), Input("fs", "value"),
    Input("fc_1", "value"), Input("fc_2", "value"), Input("filter_window", "value"),
    Input("kaiser_beta", "value"),
)
def plot_filter(filter_type, tap_count, fs, fc_1, fc_2, window, beta):
    try:
        taps = get_ftaps(filter_type, tap_count, fc_1, fc_2, fs, window, beta)
        fs = sampling_rate(fs)
    except ValueError as exc:
        logger.warning("Invalid FIR settings: %s", exc)
        blank = empty_figure("Correct the FIR settings to preview the filter")
        return blank, blank, {"error": str(exc)}, alert(str(exc))
    tap_plot = style_figure(go.Figure(go.Scatter(y=taps, mode="lines")), "FIR coefficients", "Tap", "Coefficient")
    frequencies, response = signal.freqz(taps, worN=next_fast_len(max(2048, len(taps))), fs=fs)
    magnitude = 20 * np.log10(np.maximum(np.abs(response), np.finfo(float).tiny))
    response_plot = style_figure(
        go.Figure(go.Scatter(x=frequencies, y=magnitude, mode="lines")),
        "Filter response", "Frequency (Hz)", "Magnitude (dB)",
    )
    delay = (len(taps) - 1) / (2 * fs)
    message = (
        f"Delay: {delay:g} s ({(len(taps) - 1) // 2} samples). "
        "Delay is compensated by shifting the output and shortening its end. "
        "Startup transients may remain."
    )
    feedback = compact_feedback("filter_delay", f"Delay: {delay:g} s", message)
    return tap_plot, response_plot, {"taps": taps.tolist(), "fs": fs}, feedback


def update_td_plot(mem_data, selected_var, fs, fil_val, fil_taps):
    channel = channel_data(mem_data, selected_var)
    fs = sampling_rate(fs)
    values = samples_array(channel["data"])
    figure = go.Figure(go.Scatter(
        name="Raw measurement", x=np.arange(len(values)) / fs, y=values,
        mode="lines", line=dict(color="#60a5fa"),
    ))
    filtered_data = {"filtered": None, "index": None}
    if fil_val:
        if not fil_taps or "error" in fil_taps:
            raise ValueError((fil_taps or {}).get("error", "Configure a valid FIR filter first."))
        if "fs" in fil_taps and fil_taps["fs"] != fs:
            raise ValueError("The filter is updating for the new sampling rate.")
        filtered, times, _ = apply_filter(values, fil_taps["taps"], fs)
        figure.add_trace(go.Scatter(
            name="Filtered", x=times, y=filtered, mode="lines", line=dict(color="#f87171"),
        ))
        filtered_data = {"filtered": filtered.tolist(), "index": times.tolist()}
    return style_figure(figure, "Time series", "Time (s)", channel_units(channel)), filtered_data


@app.callback(
    Output("graph_1", "figure"), Output("filt_x_mem", "data"),
    Output("signal_feedback", "children"), Output("channel_info", "children"),
    Input("memory1", "data"), Input("var_list_radio", "value"), Input("fs", "value"),
    Input("filter_apply", "value"), Input("filter_mem", "data"),
    Input("reset_selection", "n_clicks"),
)
def render_signal(memory, selected, fs, enabled, filter_data, reset):
    try:
        figure, filtered = update_td_plot(memory, selected, fs, enabled, filter_data)
        count = len(channel_data(memory, selected)["data"])
        fs = sampling_rate(fs)
    except ValueError as exc:
        if memory and selected:
            logger.warning("Cannot plot signal: %s", exc)
        return empty_figure(str(exc)), {"error": str(exc)}, alert(str(exc), "warning"), "No analysis available."
    figure.update_layout(uirevision=f"{memory['id']}:{selected}:{fs}:{reset}")
    filtered["context"] = [memory["id"], selected, fs]
    summary = (
        f"{selected} | {count:,} samples | {count / fs:g} s recording | "
        f"{fs:g} Hz sampling | Nyquist {fs / 2:g} Hz"
    )
    return figure, filtered, None, summary


@app.callback(
    Output("selection", "data"),
    Input("graph_1", "relayoutData"), Input("memory1", "data"),
    Input("var_list_radio", "value"), Input("fs", "value"),
    Input("reset_selection", "n_clicks"),
)
def update_selection(relayout, memory, selected, fs, reset):
    if ctx.triggered_id != "graph_1":
        return None
    if not relayout:
        return dash.no_update
    if relayout.get("xaxis.autorange"):
        return None
    if "xaxis.range" in relayout:
        return relayout["xaxis.range"]
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        return [relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]]
    return dash.no_update


@app.callback(
    Output("graph_2", "figure"), Output("graph_spec", "figure"),
    Output("analysis_feedback", "children"),
    Input("memory1", "data"), Input("var_list_radio", "value"), Input("fs", "value"),
    Input("nfft", "value"), Input("selection", "data"), Input("interval_mode", "value"),
    Input("filt_x_mem", "data"), Input("spectrum_mode", "value"),
    Input("spec_window", "value"), Input("spec_nfft", "value"),
    Input("spec_overlap", "value"), Input("spec_source", "value"),
)
def update_fd_plot(memory, selected, fs, nfft, selection, interval_mode, filtered_data,
                   spectrum_mode, spec_window, spec_nfft, spec_overlap, spec_source):
    try:
        channel = channel_data(memory, selected)
        fs = sampling_rate(fs)
        if not filtered_data or "error" in filtered_data:
            raise ValueError((filtered_data or {}).get("error", "Waiting for a valid signal."))
        if filtered_data.get("context") != [memory["id"], selected, fs]:
            raise ValueError("Updating analysis for the selected channel.")
        raw = samples_array(channel["data"], minimum=2)
        filtered = filtered_data["filtered"]
        count = len(raw) if filtered is None else min(len(raw), len(filtered))
        start, stop = interval_indices(count, fs, interval_mode, selection)
        units = channel_units(channel)
        labels = {
            "raw": f"Raw FFT magnitude ({units}, unnormalized)",
            "amplitude": f"Peak amplitude ({units})",
            "psd": f"PSD ({units})²/Hz",
        }
        if spectrum_mode not in labels:
            raise ValueError("Select a spectrum mode.")
        figure = go.Figure()
        series = [("Raw", raw, "#60a5fa")]
        if filtered is not None:
            series.append(("Filtered", np.asarray(filtered), "#f87171"))
        effective_nfft = number(nfft, "FFT length", minimum=2, integer=True)
        for name, samples, color in series:
            frequencies, magnitude, effective_nfft = calculate_spectrum(
                samples[start:stop], nfft, fs, spectrum_mode
            )
            figure.add_trace(go.Scatter(
                name=name, x=frequencies, y=magnitude, mode="lines", line=dict(color=color),
            ))
        style_figure(figure, "Frequency spectrum", "Frequency (Hz)", labels[spectrum_mode])
    except ValueError as exc:
        if memory and selected:
            logger.warning("Cannot analyze signal: %s", exc)
        blank = empty_figure(str(exc))
        return blank, blank, alert(str(exc), "warning")

    feedback = [alert(
        f"Analyzing {start / fs:g}–{(stop - 1) / fs:g} s | {stop - start:,} samples | "
        f"FFT length {effective_nfft:,} | bin spacing {fs / effective_nfft:g} Hz."
        + (" Raw and filtered spectra use the same available interval." if filtered is not None else ""),
        "info",
    )]
    try:
        if spec_source not in {"raw", "filtered"}:
            raise ValueError("Select a spectrogram source.")
        if spec_source == "filtered" and filtered is None:
            raise ValueError("Enable a valid FIR filter to view the filtered spectrogram.")
        source = raw if spec_source == "raw" else np.asarray(filtered)
        frequencies, times, power_db, segment = calculate_spectrogram(
            source[start:stop], fs, spec_window, spec_nfft, spec_overlap
        )
        spectrogram = style_figure(
            go.Figure(go.Heatmap(
                x=times + start / fs, y=frequencies, z=power_db, colorscale="Viridis",
                colorbar=dict(title=f"dB ({units})²/Hz"),
            )),
            f"Spectrogram ({spec_source})", "Time (s)", "Frequency (Hz)",
        )
        if segment < float(spec_window):
            feedback.append(alert(
                f"Short interval: spectrogram window reduced to {segment} samples.", "info"
            ))
    except ValueError as exc:
        logger.warning("Cannot calculate spectrogram: %s", exc)
        spectrogram = empty_figure(str(exc))
        feedback.append(alert(str(exc), "warning"))
    return figure, spectrogram, feedback


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
