import dash
import numpy as np
from left_panel import *
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from utils import parse_contents, calculate_fft, get_ftaps
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy import signal

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(
    [
        html.H4("Nyquist explorer"),
        html.Div(
            [
                html.Div(
                    id="left_panel",
                    children=[
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    files_location,
                                    html.Div(id="selected_file"),
                                    file_details,
                                ]
                            )
                        )
                    ],
                    style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'marginRight': '4px',
                           'marginLeft': '2px', 'height': '100%'}
                ),
                dcc.Store(id="memory1"),
                dcc.Store(id="filter_mem"),
                dcc.Store(id="filt_x_mem"),
                html.Div(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(label="Raw & Freq plots", children=[
                                                dcc.Graph(id="graph_1",
                                                          style={"height": "42vh", "marginBottom": "4px"}),
                                                dcc.Graph(id="graph_2", style={"height": "42vh"})
                                            ]),
                                            dbc.Tab(label="Spectrogram", children=[
                                                dcc.Graph(id="graph_spec",
                                                          style={"height": "85vh"})
                                            ]),
                                        ]
                                    )
                                ]
                            ),
                            style={"height": "92vh"},
                        )
                    ],
                    style={'width': '59%', 'height': '92vh', 'display': 'inline-block', 'vertical-align': 'top'}
                ),
                html.Div(
                    children=[
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    right_panel
                                ]
                            )
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    filter_panel
                                ]
                            ),
                        )
                    ],
                    style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'marginRight': '2px',
                           'marginLeft': '4px'}

                )
            ]
        )
    ]
)


@app.callback(
    Output("variable_list", "children"),
    [Input("memory1", "data")],
)
def update_variable_list(mem_data):
    var_list = list(mem_data['data']['data_vars'].keys())
    var_buttons = create_list_radio(var_list, "var_list_radio")
    return var_buttons


@app.callback(
    Output("metadata", "children"),
    [Input("memory1", "data")],
)
def update_metadata(mem_data):
    meta_head = pd.Series(mem_data['data']).head(5)
    print(meta_head)
    return str(meta_head)


@app.callback(
    Output("variable_content", "children"),
    [Input("var_list_radio", "value"),
     Input("memory1", "data")]
)
def update_variable_content(radio_value, mem_data):
    head_dict = pd.Series(mem_data['data']['data_vars'][radio_value]).head()
    return str(head_dict)


@app.callback(
    Output("memory1", "data"),
    Output("selected_file", "children"),
    [Input("upload", "contents"), Input("upload", "filename")],
)
def update_memory(contents, filename):
    if not contents:
        raise PreventUpdate
    file_contents = parse_contents(contents, filename).to_dict('records')
    return dict({'filenames': filename, 'data': file_contents}), f"File: {filename}"


@app.callback(
    Output("graph_1", "figure"),
    Output("filt_x_mem", "data"),
    [Input("memory1", "data"),
     Input("var_list_radio", "value"),
     Input("fs", "value"),
     Input("filter_apply", "value"),
     Input("filter_mem", "data")]
)
def update_td_plot(mem_data, selected_var, fs, fil_val, fil_taps):
    fs = int(fs)
    # nfft = int(nfft)
    var_data = mem_data['data']['data_vars'][selected_var]["data"]
    df = pd.DataFrame(
        data=var_data,
        index=np.linspace(0, len(var_data) / fs, len(var_data)),
        columns=[selected_var]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name='Raw measurement',
            x=df.index,
            y=df[selected_var],
            mode="lines",
            line=dict(
                color="blue"
            )
        )
    )
    print(fil_val, fil_taps)
    if type(fil_val) is list and len(fil_val) > 0:
        filtered = signal.lfilter(fil_taps["taps"], 1.0, df[selected_var])
        delay = 0.5 * (len(fil_taps["taps"]) - 1) / fs
        print(f"delay--{delay}")
        indices = df.index[:-int((len(fil_taps["taps"]) - 1) / 2)]
        df_filt = pd.DataFrame(data=filtered[int((len(fil_taps["taps"]) - 1) / 2):], index=indices,
                               columns=["filtered"])
        # df["filtered"] = filtered[int((len(fil_taps["taps"]) - 1)/2) :]

        print(df_filt.head(10))
        fig.add_trace(
            go.Scatter(
                name='Filtered',
                x=df_filt.index,
                y=df_filt["filtered"],
                mode="lines",
                line=dict(
                    color="red"
                )
            )
        )
        filt_x = {"filtered": filtered[int((len(fil_taps["taps"]) - 1) / 2):], "index": indices}
    else:
        filt_x = {"filtered": None, "index": None}
    fig.update_layout(xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=20, b=20),
                      legend=dict(yanchor="top", y=1, xanchor="left", x=0))
    return fig, filt_x


@app.callback(
    Output('graph_2', 'figure'),
    Output("graph_spec", "figure"),
    [Input("memory1", "data"),
     Input("var_list_radio", "value"),
     Input("fs", "value"),
     Input("nfft", "value"),
     Input('graph_1', 'relayoutData'),
     Input("filt_x_mem", "data")
     ])
def update_fd_plot(mem_data, selected_var, fs, nfft, relayoutData, filt_x):
    fs = int(fs)
    nfft = int(nfft)
    var_data = mem_data['data']['data_vars'][selected_var]["data"]
    # df = pd.DataFrame(
    #     data=var_data,
    #     index=np.linspace(0, len(var_data) / fs, len(var_data)),
    #     columns=[selected_var]
    # )
    if 'xaxis.range[0]' in relayoutData:
        start = int(relayoutData["xaxis.range[0]"] * fs)
        end = int(relayoutData["xaxis.range[1]"] * fs)
    else:
        start = 0
        end = fs
    fft_df = calculate_fft(np.array(var_data[start:end]), nfft, fs)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name='Frequency spectrum',
            x=fft_df["Frequency"],
            y=fft_df["Amplitude"],
            mode="lines",
            line=dict(
                color="blue"
            )
        )
    )
    freqs, t, Pxx = signal.spectrogram(np.array(var_data[start:end]),fs=fs,nfft=fs, window=signal.get_window("hamming", fs, fftbins=True))

    if filt_x["filtered"] is not None:
        fft_filt = calculate_fft(np.array(filt_x["filtered"][start:end]), nfft, fs)
        fig.add_trace(
            go.Scatter(
                name='Frequency spectrum (Filtered)',
                x=fft_filt["Frequency"],
                y=fft_filt["Amplitude"],
                mode="lines",
                line=dict(
                    color="red"
                )
            )
        )
        freqs, t, Pxx = signal.spectrogram(np.array(filt_x["filtered"][start:end]), fs=fs, nfft=fs, window=signal.get_window("hamming", fs, fftbins=True))

    fig.update_layout(xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=20, b=20),
                      legend=dict(yanchor="top", y=1, xanchor="left", x=0))

    # Plot with plotly
    trace = [go.Heatmap(
        x=t,
        y=freqs,
        z=10 * np.log10(Pxx),
        colorscale='Jet',
    )]
    layout = go.Layout(
        yaxis=dict(title='Frequency'),  # x-axis label
        xaxis=dict(title='Time'),  # y-axis label
    )
    fig2 = go.Figure(data=trace, layout=layout)

    return fig, fig2


@app.callback(Output("graph_3", "figure"),
              Output("graph_4", "figure"),
              Output("filter_mem", "data"),
              [Input("filter_type", "value"),
               Input("nfft", "value"),
               Input("fs", "value"),
               Input("fc_1", "value"),
               Input("fc_2", "value")])
def plot_filter(filter_type, n_fft, fs, fc_1, fc_2):
    fs = float(fs)
    fc_1 = int(fc_1)
    fc_2 = int(fc_2)
    n_fft = int(n_fft)

    y = get_ftaps(filter_type, n_fft + 1, fc_1, fc_2, fs)
    print(y)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Filter taps",
            y=y,
            mode="lines+markers",
            line=dict(
                color="blue"
            )
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False,
                      margin=dict(l=5, r=5, t=5, b=5),
                      xaxis=go.XAxis(showticklabels=False),
                      legend=dict(yanchor="top", y=1, xanchor="left", x=0))

    w, h = signal.freqz(y, worN=int(fs / 2))
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            name="Filter taps",
            x=(w / np.pi) * (fs / 2),
            y=20 * np.log10(abs(h)),
            mode="lines",
            line=dict(
                color="blue"
            )
        )
    )
    fig2.update_layout(xaxis_rangeslider_visible=False,
                       margin=dict(l=5, r=5, t=5, b=5),
                       xaxis=go.XAxis(title='Frequency'),
                       yaxis=go.XAxis(title='Magnitude (dB)')
                       )

    filter_mem_data = {
        "taps": y
    }
    return fig, fig2, filter_mem_data

if __name__ == '__main__':
    app.run_server(debug=False, host ='0.0.0.0', port = 8050)
