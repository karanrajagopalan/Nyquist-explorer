from dash import dcc, html
import dash_bootstrap_components as dbc

from constants import FILTERS


def info_tip(identifier, description, label):
    return html.Span(
        [
            html.Button(
                "i", id=f"{identifier}_info", type="button", className="settings-info",
                **{"aria-label": f"About {label}"},
            ),
            dbc.Tooltip(
                description, target=f"{identifier}_info", trigger="hover focus",
                placement="auto", className="settings-tooltip",
            ),
        ],
        className="settings-info-wrapper",
    )


def field_label(label, identifier, help_text):
    return html.Div(
        [
            dbc.Label(label, html_for=identifier),
            info_tip(identifier, help_text, label) if help_text else None,
        ],
        className="settings-label",
    )


def numeric_input(label, identifier, value, minimum=None, step="any", help_text=None):
    return html.Div(
        [
            field_label(label, identifier, help_text),
            dbc.Input(id=identifier, type="number", value=value, min=minimum,
                      step=step, debounce=True, size="sm"),
        ],
        className="settings-field",
    )


def dropdown(label, identifier, options, value, help_text=None):
    return html.Div(
        [
            field_label(label, identifier, help_text),
            dbc.Select(
                id=identifier, options=[
                    {"label": option, "value": option} if isinstance(option, str) else option
                    for option in options
                ], value=value, size="sm",
            ),
        ],
        className="settings-field",
    )


def setting_switch(label, identifier, value, help_text):
    return html.Div(
        [
            dbc.Switch(id=identifier, label=label, value=value),
            info_tip(identifier, help_text, label),
        ],
        className="settings-switch",
    )


def compact_feedback(identifier, summary, description):
    return html.Div(
        [html.Span(summary), info_tip(identifier, description, "filter design details")],
        className="settings-summary",
    )


files_location = dcc.Upload(
    id="upload",
    children=html.Div(["Drop a file here or ", html.A("choose a file")]),
    accept=".nc,.csv",
    multiple=False,
    style={
        "padding": "1rem", "border": "1px dashed #6c757d",
        "borderRadius": "0.375rem", "textAlign": "center", "cursor": "pointer",
    },
)

file_details = html.Div(
    [
        html.Div(id="upload_feedback", className="mt-2", role="status"),
        html.H6("Signal channel", className="mt-3"),
        dcc.Dropdown(id="var_list_radio", options=[], placeholder="Upload a recording", clearable=False),
        html.Div(id="variable_content", className="small mt-3", style={"overflowWrap": "anywhere"}),
        html.Details([html.Summary("File metadata"), html.Div(id="metadata", className="small mt-2")],
                     className="mt-3"),
    ]
)

right_panel = html.Div(
    [
        html.H6("Analysis", className="settings-heading"),
        numeric_input("Sample rate (Hz)", "fs", 50000, minimum=0,
                      help_text="Uses channel Fs metadata when available. You can override it."),
        dropdown("Analyze", "interval_mode", [
            {"label": "Full recording", "value": "full"},
            {"label": "Selected interval", "value": "selected"},
        ], "full", "For a selected interval, zoom horizontally in the time-series plot."),
        dbc.Button("Reset selection", id="reset_selection", n_clicks=0, size="sm",
                   color="secondary", outline=True, className="settings-reset"),
        numeric_input("Min. FFT size", "nfft", 2048, minimum=2, step=1,
                      help_text="Zero-pads shorter intervals. Longer intervals use all samples, never truncate."),
        dropdown("Spectrum", "spectrum_mode", [
            {"label": "Raw magnitude", "value": "raw"},
            {"label": "Amplitude (peak)", "value": "amplitude"},
            {"label": "PSD", "value": "psd"},
        ], "raw", "Raw: unnormalized FFT magnitude. Amplitude: single-sided peak amplitude. "
           "PSD: power spectral density in signal-units squared per Hz."),
        html.H6("Spectrogram", className="settings-heading"),
        numeric_input("Window samples", "spec_window", 256, minimum=2, step=1,
                      help_text="Hamming window length in samples. Short intervals reduce this automatically."),
        numeric_input("FFT size", "spec_nfft", 512, minimum=2, step=1,
                      help_text="Independent of the spectrum FFT. Must be at least the requested window length."),
        numeric_input("Overlap (%)", "spec_overlap", 50, minimum=0,
                      help_text="Percentage shared by neighboring windows; must be below 100%."),
        dropdown("Source", "spec_source", [
            {"label": "Raw signal", "value": "raw"},
            {"label": "Filtered signal", "value": "filtered"},
        ], "raw", "Choose raw samples or enable the FIR filter to analyze its output."),
    ]
)

filter_panel = html.Div(
    [
        html.H6("FIR filter", className="settings-heading"),
        setting_switch("Apply filter", "filter_apply", False,
                       "Overlay the filtered signal and compare spectra over the same available interval."),
        dropdown("Type", "filter_type", [
            {"label": "Low-pass", "value": "lowpass"},
            {"label": "High-pass", "value": "highpass"},
            {"label": "Band-pass", "value": "bandpass"},
        ], "lowpass", "Low-pass keeps lower frequencies; high-pass keeps higher frequencies; "
           "band-pass keeps the band between both cutoffs."),
        numeric_input("Cutoff (Hz)", "fc_1", 500, minimum=0,
                      help_text="Cutoff, or lower cutoff for band-pass. Must be strictly between zero and Nyquist."),
        html.Div(numeric_input("Upper cutoff (Hz)", "fc_2", 2000, minimum=0,
                               help_text="Band-pass upper edge. Must exceed the lower cutoff and remain below Nyquist."),
                 id="upper_cutoff", style={"display": "none"}),
        setting_switch("Auto tap count", "auto_taps", True,
                       "Kaiser estimate targeting 60 dB attenuation. Transition width is 10% of the "
                       "smallest cutoff-to-edge or band-pass span. Turn off to edit taps, window and beta."),
        html.Div(id="auto_taps_feedback", role="status"),
        numeric_input("Taps", "tap_count", 101, minimum=3, step=1,
                      help_text="Order = taps - 1. Turn off automatic length to edit taps and window."),
        dropdown("Window", "filter_window", FILTERS, "kaiser",
                 "FIR design window. Automatic tap count uses Kaiser; disable it to choose another window."),
        html.Div(numeric_input("Kaiser beta", "kaiser_beta", 8.6, minimum=0,
                               help_text="Kaiser window shape parameter. Set automatically for the 60 dB estimate; editable in manual mode."),
                 id="kaiser_settings", style={"display": "none"}),
        html.Div(id="filter_feedback", role="status"),
        html.Details(
            [
                html.Summary("Filter preview"),
                dbc.Tabs(
                    [
                        dbc.Tab(dcc.Graph(id="graph_4", responsive=True, className="filter-graph"),
                                label="Response", tab_id="response"),
                        dbc.Tab(dcc.Graph(id="graph_3", responsive=True, className="filter-graph"),
                                label="Taps", tab_id="taps"),
                    ],
                    active_tab="response",
                ),
            ],
            className="filter-preview",
        ),
    ]
)
