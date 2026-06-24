import os
import glob

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from predict import predict_npz
from explain_prediction import generate_explanation

app = dash.Dash(__name__)
server = app.server

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
                html.Th("Scientific Score"),
                html.Th("Period"),
                html.Th("Duration"),
                html.Th("Depth"),
                html.Th("SNR")
            ])
        ]

        +

        [
            html.Tr([
                html.Td(row["rank"]),
                html.Td(row["file"]),
                html.Td(row["prediction"]),
                html.Td(round(row["confidence"],4)),
                html.Td(round(row["scientific_score"],4)),
                html.Td(round(row["period_days"],2)),
                html.Td(round(row["duration_hours"],2)),
                html.Td(round(row["depth_ppm"],2)),
                html.Td(round(row["snr"],2))
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
        Output("prediction-output", "children"),
        Output("global-view-graph", "figure"),
        Output("local-view-graph", "figure")
    ],
    [
        Input("candidate-dropdown", "value")
    ]
)
def update_candidate(path):

    filename = os.path.basename(path)

    row = ranking_df[
        ranking_df["file"] == filename
    ].iloc[0]

    data = np.load(path, allow_pickle=True)

    explanations = []

    if result["confidence"] > 0.90:
        explanations.append(
            "Model confidence is very high."
        )
    elif result["confidence"] > 0.75:
        explanations.append(
            "Model confidence is moderate."
        )
    else:
        explanations.append(
            "Model confidence is low."
        )

    if result["snr"] > 20:
        explanations.append(
            "Strong signal-to-noise ratio."
        )
    elif result["snr"] > 10:
        explanations.append(
            "Acceptable signal-to-noise ratio."
        )
    else:
        explanations.append(
            "Weak signal-to-noise ratio."
        )

    if result["depth_ppm"] > 500:
        explanations.append(
            "Transit depth is clearly visible."
        )
    else:
        explanations.append(
            "Transit depth is relatively shallow."
        )

    if result["duration_hours"] < 10:
        explanations.append(
            "Transit duration is consistent with many planetary candidates."
        )
    else:
        explanations.append(
            "Transit duration is unusually long."
        )

    if result["period_days"] > 30:
        explanations.append(
            "Candidate has a relatively long orbital period."
        )
    elif result["period_days"] > 10:
        explanations.append(
            "Candidate has a moderate orbital period."
        )
    else:
        explanations.append(
            "Candidate has a short orbital period."
        )

    if result["scientific_score"] > 0.80:
        explanations.append(
            "High-priority candidate for follow-up study."
        )
    elif result["scientific_score"] > 0.60:
        explanations.append(
            "Moderately interesting candidate."
        )
    else:
        explanations.append(
            "Low-priority candidate."
        )

    global_fig = go.Figure()

    global_fig.add_trace(
        go.Scatter(
            y=data["global_view"],
            mode="lines"
        )
    )

    global_fig.update_layout(
        title="Global View (201 bins)"
    )

    local_fig = go.Figure()

    local_fig.add_trace(
        go.Scatter(
            y=data["local_view"],
            mode="lines"
        )
    )

    local_fig.update_layout(
        title="Local View (61 bins)"
    )

    pred_text = html.Div([

        html.H3(
            f"Prediction: {row['prediction']}"
        ),

        html.H4(
            f"Confidence: {row['confidence']:.4f}"
        ),

        html.H4(
            f"Scientific Score: {row['scientific_score']:.4f}"
        ),

        html.H4(
            f"Period: {row['period_days']:.4f} days"
        ),

        html.H4(
            f"Duration: {row['duration_hours']:.4f} hours"
        ),

        html.H4(
            f"Depth: {row['depth_ppm']:.2f} ppm"
        ),

        html.H4(
            f"SNR: {row['snr']:.2f}"
        ),

        html.Hr(),

        html.H3(
            "Prediction Explanation"
        ),

        html.Ul([
            html.Li(x)
            for x in explanations
        ])
    ])

    return (
        pred_text,
        global_fig,
        local_fig
    )

if __name__ == "__main__":
    app.run(debug=True)