from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from constants import *

files_location = dcc.Upload(
    id="upload",
    children=[
        'Drag and Drop or ',
        html.A('Select a File')
    ],
    multiple=False,

    style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center'
    }
)


def create_files_list(files):
    acc_files = []
    for i, file in enumerate(files):
        acc_files.append(
            dbc.AccordionItem(
                title=file
            )
        )
    accordion = html.Div(
        dbc.Accordion(
            acc_files,
            start_collapsed=True,
        ),
    )
    return accordion


def create_list_radio(files, id_name):
    radio_options = []
    for i, file in enumerate(files):
        radio_options.append(
            {"label": file, "value": file}
        )
    radio = dbc.RadioItems(
        options=radio_options,
        value=files[0],
        id=id_name
    )
    return radio


def create_input_box(name, id_1, ph, value):
    return dbc.Row(
        [
            dbc.Col(html.Div(name)),
            dbc.Col(dbc.Input(id=id_1, placeholder=ph, value=value)),
            dbc.Row()
        ]
    )


def create_drop_down(name, op_list, id_1, default_value=0):
    return dbc.Row(
        [
            dbc.Col(html.Div(name)),
            dbc.Col(
                dcc.Dropdown(
                    id=id_1,
                    options=[dict(label=x, value=x) for x in op_list],
                    value=op_list[default_value]
                )
            ),
            dbc.Row()
        ]
    )


def create_checkbox(name, op_list, id_1, default_value=0, switch_flag=False):
    return dbc.Row(
        [
            dbc.Col(dcc.Markdown(f""" ** {name} ** """)),
            # dbc.Col(html.Div(name)),
            dbc.Col(
                dbc.Checklist(
                    id=id_1,
                    options=[dict(value=x) for x in op_list],
                    value=op_list[default_value],
                    switch=switch_flag
                )
            ),
            dbc.Row()
        ]
    )


file_details = html.Div(
    children=[
        html.H6("File contents"),
        html.Div(
            id="metadata",
            children=[],
            style={"height": "24vh", "maxHeight": "24vh", "overflow": "scroll"}
        ),
        html.H6("Data variables"),
        html.Div(
            id="variable_list",
            children=[],
            style={"height": "24vh", "maxHeight": "24vh", "overflow": "scroll"},
        ),
        html.H6("Variable contents"),
        html.Div(
            id="variable_content",
            children=[],
            style={"height": "24vh", "maxHeight": "24vh", "overflow": "scroll"},
        ),
    ],
    id="File_contents",
    style={"marginTop": "2px", "marginBottom": "2px"}
)

right_panel = html.Div(
    [
        create_input_box("Sampling Frequency---->", "fs", "50000", 50000),
        create_input_box("NFFT--------->", "nfft", "2048", 2048),
    ]
)

filter_panel = html.Div(
    [
        create_checkbox("Filter:", [0], "filter_apply", 0, True),
        create_drop_down("Window type", FILTERS, "filter_type", 11),
        create_input_box("Cut_off_1", "fc_1", "0", 0),
        create_input_box("Cut_off_2", "fc_2", "500", 500),
        dcc.Graph(id="graph_3", style={"height": "29vh"}),
        dcc.Graph(id="graph_4", style={"height": "29vh"})
    ],
)
