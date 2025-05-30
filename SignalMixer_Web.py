import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# --- Neon Colors (Hex for Plotly) ---
NEON_BLUE = '#00FFFF'
NEON_GREEN = '#39FF14'
NEON_RED = '#FF073A'
PLOTLY_DARK_BG = '#111111'
PLOTLY_PAPER_BG = '#1e1e1e'
PLOTLY_GRID_COLOR = '#444444'
PLOTLY_FONT_COLOR = 'lightgray'

# --- Constants and Initial Parameters ---
TWO_PI = 2 * np.pi
common_initial_A = 1.0; common_initial_f = 4.0; common_initial_phi_deg = 0.0
default_A1 = common_initial_A; default_f1 = common_initial_f; default_phi1_deg = common_initial_phi_deg
default_A2 = common_initial_A; default_f2 = common_initial_f; default_phi2_deg = common_initial_phi_deg
initial_y_separation_offset = 0.0
t = np.linspace(0, 2, 2000)

# --- Dynamic Line Width (Simulating Glow) Parameters ---
INITIAL_LINE_WIDTH_FACTOR = 2.0  # MODIFIED: Halved for 50% reduction in initial thickness
MIN_LINE_WIDTH_FACTOR = 1.2      # Min line width factor at high frequency
CORE_LINE_WIDTH_W1_W2 = 1.5
CORE_LINE_WIDTH_SUM = 2.0
MAX_FREQ_SLIDER_VALUE = 15.0

# --- Helper Functions ---
def sine_wave(amplitude, frequency, phase_radians, time_vector):
    if frequency == 0:
        return amplitude * np.sin(phase_radians) * np.ones_like(time_vector)
    return amplitude * np.sin(TWO_PI * frequency * time_vector + phase_radians)

def calculate_dynamic_line_width(current_frequency, max_frequency,
                                 initial_factor=INITIAL_LINE_WIDTH_FACTOR,
                                 min_factor=MIN_LINE_WIDTH_FACTOR,
                                 core_lw=1.5):
    if max_frequency <= 0:
        return core_lw * initial_factor # Max width if no valid max_frequency
        
    norm_freq = min(abs(current_frequency) / max_frequency, 1.0)
    
    # The factor itself is interpolated (from initial_factor down to min_factor)
    interpolated_factor = initial_factor - norm_freq * (initial_factor - min_factor)
    
    # The final line width is the core line width multiplied by this interpolated factor
    # Ensure the interpolated_factor doesn't go below what min_factor implies
    # This means min_factor is the lowest *multiplier* for core_lw
    final_width = core_lw * max(interpolated_factor, min_factor) 
    return final_width


# --- Dash App Initialization ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- App Layout ---
app.layout = html.Div(style={'backgroundColor': PLOTLY_DARK_BG, 'color': PLOTLY_FONT_COLOR, 'fontFamily': 'Arial', 'height': '100vh', 'display': 'flex', 'flexDirection': 'column'}, children=[
    html.H1("UTE - Introduction to CS - Interactive Signal Viewer", style={'textAlign': 'center', 'padding': '10px 0', 'margin': '0', 'flexShrink': 0, 'fontSize': '24px'}),
    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'flexGrow': 1, 'overflow': 'hidden'}, children=[
        html.Div(
            style={'flex': '0 0 300px', 'padding': '15px', 'backgroundColor': PLOTLY_PAPER_BG, 'borderRadius': '5px', 'margin': '10px', 'overflowY': 'auto', 'height': 'calc(100% - 20px)'},
            children=[
                html.H3("Wave Parameters", style={'marginTop': 0}),
                html.H4("Wave 1", style={'color': NEON_BLUE, 'marginBottom': '5px'}),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}, children=[
                    html.Label("A₁:", style={'minWidth': '70px', 'marginRight': '10px'}),
                    html.Div(dcc.Slider(id='slider-A1', min=0, max=18, step=1, value=default_A1, marks=None, tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'), style={'flex': '1'})
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}, children=[
                    html.Label("f₁ (Hz):", style={'minWidth': '70px', 'marginRight': '10px'}),
                    html.Div(dcc.Slider(id='slider-f1', min=0, max=MAX_FREQ_SLIDER_VALUE, step=1, value=default_f1, marks=None, tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'), style={'flex': '1'})
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}, children=[
                    html.Label("φ₁ (deg):", style={'minWidth': '70px', 'marginRight': '10px'}),
                    html.Div(dcc.Slider(id='slider-phi1', min=0, max=360, step=10, value=default_phi1_deg, marks=None, tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'), style={'flex': '1'})
                ]),
                html.Hr(),
                html.H4("Wave 2", style={'color': NEON_GREEN, 'marginBottom': '5px'}),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}, children=[
                    html.Label("A₂:", style={'minWidth': '70px', 'marginRight': '10px'}),
                    html.Div(dcc.Slider(id='slider-A2', min=0, max=18, step=1, value=default_A2, marks=None, tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'), style={'flex': '1'})
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}, children=[
                    html.Label("f₂ (Hz):", style={'minWidth': '70px', 'marginRight': '10px'}),
                    html.Div(dcc.Slider(id='slider-f2', min=0, max=MAX_FREQ_SLIDER_VALUE, step=1, value=default_f2, marks=None, tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'), style={'flex': '1'})
                ]),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}, children=[
                    html.Label("φ₂ (deg):", style={'minWidth': '70px', 'marginRight': '10px'}),
                    html.Div(dcc.Slider(id='slider-phi2', min=0, max=360, step=10, value=default_phi2_deg, marks=None, tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'), style={'flex': '1'})
                ]),
                html.Hr(),
                html.H3("Controls"),
                html.Label("Operation:"),
                dcc.RadioItems(id='radio-operation', options=[{'label': 'Add', 'value': 'Add'}, {'label': 'Multiply', 'value': 'Multiply'}], value='Add', labelStyle={'display': 'inline-block', 'marginRight': '10px'}),
                html.Br(),
                html.Label("View Mode:"),
                dcc.RadioItems(id='radio-view-mode', options=[{'label': 'Combined', 'value': 'Combined'}, {'label': 'Separate', 'value': 'Separate'}], value='Combined', labelStyle={'display': 'inline-block', 'marginRight': '10px'}),
                html.Div(id='y-offset-slider-div', children=[
                    html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'}, children=[
                        html.Label("Y-Offset:", style={'minWidth': '70px', 'marginRight': '10px'}),
                        html.Div(dcc.Slider(id='slider-y-offset', min=0, max=30, step=0.1, value=initial_y_separation_offset, marks=None, tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'), style={'flex': '1'})
                    ]),
                ]),
                html.Div(id='visibility-checkboxes-div', style={'marginTop': '10px'}, children=[
                    html.Label("Visibility (Combined View):"),
                    dcc.Checklist(id='checklist-visibility', options=[{'label': 'Show Wave 1', 'value': 'w1'}, {'label': 'Show Wave 2', 'value': 'w2'}, {'label': 'Show Result', 'value': 'sum'}], value=['w1', 'w2', 'sum'], labelStyle={'display': 'block'})
                ]),
                html.Br(),
                html.Button('Reset Waves', id='button-reset', n_clicks=0, style={'marginTop': '10px', 'width': '100%'}),
                html.Hr(),
                html.Div(id='equations-display', style={'marginTop': '15px', 'fontSize': '12px', 'lineHeight': '1.6'})
            ]
        ),
        html.Div(
            style={'flex': '1', 'padding': '10px', 'display': 'flex', 'flexDirection': 'column', 'height': 'calc(100% - 20px)'},
            children=[
                dcc.Graph(id='main-graph', style={'flex': '1'}, config={'responsive': True})
            ]
        )
    ])
])

# --- Callbacks ---
@app.callback(
    [Output('y-offset-slider-div', 'style'),
     Output('visibility-checkboxes-div', 'style')],
    [Input('radio-view-mode', 'value')]
)
def toggle_combined_view_controls(view_mode):
    if view_mode == 'Combined':
        return {'display': 'block'}, {'display': 'block', 'marginTop': '10px'}
    else:
        return {'display': 'none'}, {'display': 'none'}

@app.callback(
    [Output('slider-A1', 'value'), Output('slider-f1', 'value'), Output('slider-phi1', 'value'),
     Output('slider-A2', 'value'), Output('slider-f2', 'value'), Output('slider-phi2', 'value'),
     Output('slider-y-offset', 'value'),
     Output('radio-operation', 'value'),
     Output('radio-view-mode', 'value'),
     Output('checklist-visibility', 'value')],
    [Input('button-reset', 'n_clicks')],
    prevent_initial_call=True
)
def reset_parameters(n_clicks):
    return (default_A1, default_f1, default_phi1_deg,
            default_A2, default_f2, default_phi2_deg,
            initial_y_separation_offset,
            'Add', 'Combined', ['w1', 'w2', 'sum'])

@app.callback(
    [Output('main-graph', 'figure'),
     Output('equations-display', 'children')],
    [Input('slider-A1', 'value'), Input('slider-f1', 'value'), Input('slider-phi1', 'value'),
     Input('slider-A2', 'value'), Input('slider-f2', 'value'), Input('slider-phi2', 'value'),
     Input('slider-y-offset', 'value'),
     Input('radio-operation', 'value'),
     Input('radio-view-mode', 'value'),
     Input('checklist-visibility', 'value')]
)
def update_graph_and_equations(A1, f1, phi1_deg, A2, f2, phi2_deg,
                               y_offset,
                               operation, view_mode, visibility):
    A1 = A1 if A1 is not None else default_A1
    f1 = f1 if f1 is not None else default_f1
    phi1_deg = phi1_deg if phi1_deg is not None else default_phi1_deg
    A2 = A2 if A2 is not None else default_A2
    f2 = f2 if f2 is not None else default_f2
    phi2_deg = phi2_deg if phi2_deg is not None else default_phi2_deg
    y_offset = y_offset if y_offset is not None else initial_y_separation_offset

    phi1_rad = np.deg2rad(phi1_deg)
    phi2_rad = np.deg2rad(phi2_deg)
    y1_orig = sine_wave(A1, f1, phi1_rad, t)
    y2_orig = sine_wave(A2, f2, phi2_rad, t)

    if operation == "Add":
        ysum_orig = y1_orig + y2_orig
        sum_op_text = "y₁ + y₂"
    else:
        ysum_orig = y1_orig * y2_orig
        sum_op_text = "y₁ ⋅ y₂"

    eq_md = f"""
    $y_1 = {A1:.1f} \sin(2\pi {f1:.1f} t + {phi1_deg:.0f}^\circ)$
    <br>
    $y_2 = {A2:.1f} \sin(2\pi {f2:.1f} t + {phi2_deg:.0f}^\circ)$
    <br>
    Result: ${sum_op_text}$
    """
    equations_display_children = [dcc.Markdown(eq_md, dangerously_allow_html=True, mathjax=True)]

    lw1 = calculate_dynamic_line_width(f1, MAX_FREQ_SLIDER_VALUE, core_lw=CORE_LINE_WIDTH_W1_W2)
    lw2 = calculate_dynamic_line_width(f2, MAX_FREQ_SLIDER_VALUE, core_lw=CORE_LINE_WIDTH_W1_W2)
    effective_sum_freq = max(abs(f1), abs(f2)) if f1 != 0 or f2 != 0 else 0
    lw_sum = calculate_dynamic_line_width(effective_sum_freq, MAX_FREQ_SLIDER_VALUE, core_lw=CORE_LINE_WIDTH_SUM)

    fig = go.Figure()
    plot_layout = {
        'template': 'plotly_dark',
        'paper_bgcolor': PLOTLY_DARK_BG,
        'plot_bgcolor': PLOTLY_DARK_BG,
        'font_color': PLOTLY_FONT_COLOR,
        'autosize': True,
        'xaxis': {'gridcolor': PLOTLY_GRID_COLOR, 'zerolinecolor': PLOTLY_GRID_COLOR, 'title_font': {'color': PLOTLY_FONT_COLOR}, 'tickfont': {'color': PLOTLY_FONT_COLOR}},
        'yaxis': {'gridcolor': PLOTLY_GRID_COLOR, 'zerolinecolor': PLOTLY_GRID_COLOR, 'title_font': {'color': PLOTLY_FONT_COLOR}, 'tickfont': {'color': PLOTLY_FONT_COLOR}},
        'margin': dict(l=50, r=20, t=40, b=40, autoexpand=True),
        'legend': {'bgcolor': 'rgba(30,30,30,0.8)', 'bordercolor': 'gray', 'font': {'color': PLOTLY_FONT_COLOR}},
        'title_font': {'color': PLOTLY_FONT_COLOR}
    }

    if view_mode == 'Separate':
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Wave 1", "Wave 2", "Result"))
        fig.update_layout(**plot_layout)

        fig.add_trace(go.Scatter(x=t, y=y1_orig, name='Wave 1', line=dict(color=NEON_BLUE, width=lw1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=y2_orig, name='Wave 2', line=dict(color=NEON_GREEN, width=lw2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=ysum_orig, name='Result', line=dict(color=NEON_RED, width=lw_sum)), row=3, col=1)

        all_sep_data = np.concatenate([d for d in [y1_orig, y2_orig, ysum_orig] if d is not None and len(d)>0])
        if len(all_sep_data) > 0:
            min_y_s, max_y_s = np.min(all_sep_data), np.max(all_sep_data)
            padding_s = max((max_y_s - min_y_s) * 0.1, 0.5)
            common_yrange = [min_y_s - padding_s, max_y_s + padding_s]
        else:
            common_yrange = [-1,1]

        fig.update_yaxes(title_text="Amplitude", range=common_yrange, showticklabels=True, row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", range=common_yrange, showticklabels=True, row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", range=common_yrange, showticklabels=True, row=3, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        for i in range(1, 4):
            if i-1 < len(fig.layout.annotations):
                 fig.layout.annotations[i-1].update(font=dict(color=PLOTLY_FONT_COLOR, size=14), yanchor='top', y=fig.layout.annotations[i-1].y + 0.02)
    else: # Combined view
        fig.update_layout(**plot_layout)
        fig.update_layout(title_text="Combined Waveforms", title_x=0.5,
                          xaxis_title="Time (s)", yaxis_title="Amplitude")

        y1_display = y1_orig + y_offset
        y2_display = y2_orig
        ysum_display = ysum_orig - y_offset

        if 'w1' in visibility:
            fig.add_trace(go.Scatter(x=t, y=y1_display, name='Wave 1', line=dict(color=NEON_BLUE, width=lw1)))
        if 'w2' in visibility:
            fig.add_trace(go.Scatter(x=t, y=y2_display, name='Wave 2', line=dict(color=NEON_GREEN, width=lw2)))
        if 'sum' in visibility:
            fig.add_trace(go.Scatter(x=t, y=ysum_display, name='Result', line=dict(color=NEON_RED, width=lw_sum)))

        all_visible_data = []
        if 'w1' in visibility and y1_display is not None: all_visible_data.append(y1_display)
        if 'w2' in visibility and y2_display is not None: all_visible_data.append(y2_display)
        if 'sum' in visibility and ysum_display is not None: all_visible_data.append(ysum_display)

        if all_visible_data:
            concatenated_data = np.concatenate([d for d in all_visible_data if d is not None and len(d)>0])
            if len(concatenated_data) > 0:
                min_y_c, max_y_c = np.min(concatenated_data), np.max(concatenated_data)
                padding_c = max((max_y_c - min_y_c) * 0.1, 0.5)
                fig.update_yaxes(range=[min_y_c - padding_c, max_y_c + padding_c])
            else:
                fig.update_yaxes(range=[-1,1])
        else:
            fig.update_yaxes(range=[-1, 1])

        if y_offset != 0.0:
            fig.update_yaxes(showticklabels=False)
        else:
            fig.update_yaxes(showticklabels=True)

    return fig, equations_display_children

# --- Run the app ---
if __name__ == '__main__':
    app.run(debug=True)