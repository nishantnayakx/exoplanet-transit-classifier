import os
import glob

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from predict import predict_npz

app = dash.Dash(__name__)

DATA_DIR = "data/processed"

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))

metrics = pd.DataFrame({
    "Metric": [
        "Validation Accuracy",
        "ROC-AUC",
        "Planet Precision",
        "Planet Recall",
        "Planet F1"
    ],
    "Value": [
        0.787,
        0.8424,
        0.82,
        0.87,
        0.84
    ]
})

metric_fig = px.bar(
    metrics,
    x="Metric",
    y="Value",
    title="Model Performance"
)

app.layout = html.Div([

    html.H1("Exoplanet Transit Classifier"),

    html.Hr(),

    html.H2("Dataset Overview"),

    html.Ul([
        html.Li("Total Samples: 943"),
        html.Li("Planet Transits: 593"),
        html.Li("False Positives: 350")
    ]),

    html.Hr(),

    html.H2("Model Metrics"),

    dcc.Graph(figure=metric_fig),

    html.Hr(),

    html.H2("Confusion Matrix"),

    html.Img(
        src="/assets/confusion_matrix.png",
        style={"width": "600px"}
    ),

    html.Hr(),

    html.H2("Candidate Explorer"),

    dcc.Dropdown(
        id="candidate-dropdown",
        options=[
            {
                "label": os.path.basename(f),
                "value": f
            }
            for f in files
        ],
        value=files[0]
    ),

    html.Br(),

    html.Div(id="prediction-output"),

    dcc.Graph(id="global-view-graph"),

    dcc.Graph(id="local-view-graph")
])


@app.callback(
    [
        Output("prediction-output", "children"),
        Output("global-view-graph", "figure"),
        Output("local-view-graph", "figure")
    ],
    [
        Input("candidate-dropdown", "value")
    ]
)
def update_candidate(path):

    result = predict_npz(path)

    global_fig = go.Figure()

    global_fig.add_trace(
        go.Scatter(
            y=result["global_view"],
            mode="lines",
            name="Global View"
        )
    )

    global_fig.update_layout(
        title="Global View (201 bins)"
    )

    local_fig = go.Figure()

    local_fig.add_trace(
        go.Scatter(
            y=result["local_view"],
            mode="lines",
            name="Local View"
        )
    )

    local_fig.update_layout(
        title="Local View (61 bins)"
    )

    pred_text = html.Div([
        html.H3(
            f"Prediction: {result['prediction']}"
        ),
        html.H4(
            f"Confidence: {result['confidence']:.4f}"
        )
    ])

    return pred_text, global_fig, local_fig


if __name__ == "__main__":
    app.run(debug=True)