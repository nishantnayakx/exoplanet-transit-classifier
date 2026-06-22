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

files = sorted(
    glob.glob(
        os.path.join(DATA_DIR, "*.npz")
    )
)

ranking_df = pd.read_csv(
    "candidate_ranking.csv"
)

top20_df = ranking_df.head(20)

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

prediction_fig = px.pie(
    ranking_df,
    names="prediction",
    title="Prediction Distribution"
)

confidence_fig = px.histogram(
    ranking_df,
    x="confidence",
    nbins=30,
    title="Confidence Distribution"
)

scientific_fig = px.bar(
    ranking_df.head(20),
    x="file",
    y="scientific_score",
    title="Top Scientific Candidates",
)

app.layout = html.Div([

    html.H1(
        "Exoplanet Transit Classifier Dashboard"
    ),

    html.Hr(),

    html.H2("Dataset Overview"),

    html.Ul([
        html.Li(
            f"Total Samples: {len(ranking_df)}"
        ),
        html.Li(
            f"Planet Transit Predictions: {(ranking_df['prediction']=='planet_transit').sum()}"
        ),
        html.Li(
            f"False Positive Predictions: {(ranking_df['prediction']=='false_positive').sum()}"
        )
    ]),

    html.Hr(),

    html.H2("Model Metrics"),

    dcc.Graph(
        figure=metric_fig
    ),

    html.Hr(),

    html.H2("Prediction Distribution"),

    dcc.Graph(
        figure=prediction_fig
    ),

    html.Hr(),

    html.H2("Confidence Distribution"),

    dcc.Graph(
        figure=confidence_fig
    ),

    html.Hr(),

    html.H2("Top Scientific Candidates"),

    dcc.Graph(
        figure=scientific_fig
    ),

    html.Hr(),

    html.H2("Top 20 Ranked Candidates"),

    html.Table(

        [
            html.Tr([
                html.Th("Rank"),
                html.Th("File"),
                html.Th("Prediction"),
                html.Th("Confidence"),
                html.Th("Scientific Score")
            ])
        ]

        +

        [
            html.Tr([
                html.Td(row["rank"]),
                html.Td(row["file"]),
                html.Td(row["prediction"]),
                html.Td(
                    round(
                        row["confidence"],
                        4
                    )
                ),
                html.Td(
                    round(
                        row["scientific_score"],
                        4
                    )
                )
            ])

            for _, row in top20_df.iterrows()
        ]
    ),

    html.Hr(),

    html.H2("Confusion Matrix"),

    html.Img(
        src="/assets/confusion_matrix.png",
        style={
            "width": "700px"
        }
    ),

    html.Hr(),

    html.H2("ROC Curve"),

    html.Img(
        src="/assets/roc_curve.png",
        style={
            "width": "700px"
        }
    ),

    html.Hr(),

    html.H2("Precision Recall Curve"),

    html.Img(
        src="/assets/pr_curve.png",
        style={
            "width": "700px"
        }
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

    html.Div(
        id="prediction-output"
    ),

    html.Br(),

    dcc.Graph(
        id="global-view-graph"
    ),

    dcc.Graph(
        id="local-view-graph"
    )

])


@app.callback(
    [
        Output(
            "prediction-output",
            "children"
        ),

        Output(
            "global-view-graph",
            "figure"
        ),

        Output(
            "local-view-graph",
            "figure"
        )
    ],

    [
        Input(
            "candidate-dropdown",
            "value"
        )
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
        ),

        html.H4(
            f"Period: {result['period_days']:.4f} days"
        ),

        html.H4(
            f"Duration: {result['duration_hours']:.4f} hours"
        ),

        html.H4(
            f"Depth: {result['depth_ppm']:.2f} ppm"
        ),

        html.H4(
            f"SNR: {result['snr']:.2f}"
        )

    ])

    return (
        pred_text,
        global_fig,
        local_fig
    )


if __name__ == "__main__":
    app.run(debug=True)