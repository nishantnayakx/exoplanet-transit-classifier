import os
import glob

import dash
from dash import dcc, html, Input, Output, dash_table
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

ranking_fig = px.bar(
    top20_df.head(10),
    x="scientific_score",
    y="file",
    orientation="h",
    title="Top 10 Scientific Candidates"
)

ranking_fig.update_layout(
    yaxis={"categoryorder": "total ascending"}
)

scatter_fig = px.scatter(
    ranking_df,
    x="confidence",
    y="scientific_score",
    color="prediction",
    hover_data=[
        "file",
        "period_days",
        "duration_hours",
        "depth_ppm",
        "snr"
    ],
    title="Confidence vs Scientific Score"
)

scatter_fig.update_layout(
    xaxis_title="Prediction Confidence",
    yaxis_title="Scientific Score"
)

scientific_fig = px.bar(
    ranking_df.head(20),
    x="file",
    y="scientific_score",
    title="Top Scientific Candidates",
)

CARD_STYLE = {
    "padding": "15px",
    "border": "1px solid #ddd",
    "borderRadius": "10px",
    "textAlign": "center",
    "width": "180px",
    "backgroundColor": "#f8f9fa",
    "boxShadow": "0px 2px 4px rgba(0,0,0,0.08)"
}

app.layout = html.Div([

        html.Div([

        html.Div([
            html.H3("943"),
            html.P("Total Samples")
        ], style={
            "padding": "20px",
            "border": "1px solid #ddd",
            "borderRadius": "10px",
            "textAlign": "center",
            "width": "220px"
        }),

        html.Div([
            html.H3("593"),
            html.P("Planet Transits")
        ], style={
            "padding": "20px",
            "border": "1px solid #ddd",
            "borderRadius": "10px",
            "textAlign": "center",
            "width": "220px"
        }),

        html.Div([
            html.H3("350"),
            html.P("False Positives")
        ], style={
            "padding": "20px",
            "border": "1px solid #ddd",
            "borderRadius": "10px",
            "textAlign": "center",
            "width": "220px"
        }),

        html.Div([
            html.H3("0.8424"),
            html.P("ROC-AUC")
        ], style={
            "padding": "20px",
            "border": "1px solid #ddd",
            "borderRadius": "10px",
            "textAlign": "center",
            "width": "220px"
        }),

    ], style={
        "display": "flex",
        "gap": "20px",
        "justifyContent": "center",
        "marginBottom": "30px"
    }),

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

    html.H2("Top Ranked Candidates"),

    dcc.Graph(
    figure=ranking_fig
    ),

    html.H2("Scientific Candidate Analysis"),

    dcc.Graph(
        figure=scatter_fig
    ),

    html.Button(
        "Download Candidate Rankings CSV",
        id="download-btn",
        style={
            "marginBottom": "20px"
        }
    ),

    dcc.Download(
        id="download-csv"
    ),

    dash_table.DataTable(

        data=top20_df.to_dict("records"),

        columns=[
            {"name": i, "id": i}
            for i in top20_df.columns
        ],

        page_size=20,

        sort_action="native",

        filter_action="native",

        style_table={
            "overflowX": "auto"
        },

        style_cell={
            "textAlign": "left",
            "padding": "8px"
        },

        style_header={
            "fontWeight": "bold"
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

        value=files[0],
        searchable=True,
        placeholder="Search candidate..."
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

    html.H2("System Architecture"),

    html.Img(
        src="/assets/architecture_diagram.png",
        style={
            "width": "1000px"
        }
    ),

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

    try:

        result = predict_npz(path)
        
        filename = os.path.basename(path)

        row = ranking_df[
            ranking_df["file"] == filename
        ].iloc[0]


        explanations = []

        if row["confidence"] > 0.90:
            explanations.append(
                "Model confidence is very high."
            )
        elif row["confidence"] > 0.75:
            explanations.append(
                "Model confidence is moderate."
            )
        else:
            explanations.append(
                "Model confidence is low."
            )

        if row["snr"] > 20:
            explanations.append(
                "Strong signal-to-noise ratio."
            )
        elif row["snr"] > 10:
            explanations.append(
                "Acceptable signal-to-noise ratio."
            )
        else:
            explanations.append(
                "Weak signal-to-noise ratio."
            )

        if row["depth_ppm"] > 500:
            explanations.append(
                "Transit depth is clearly visible."
            )
        else:
            explanations.append(
                "Transit depth is relatively shallow."
            )

        if row["duration_hours"] < 10:
            explanations.append(
                "Transit duration is consistent with many planetary candidates."
            )
        else:
            explanations.append(
                "Transit duration is unusually long."
            )

        if row["period_days"] > 30:
            explanations.append(
                "Candidate has a relatively long orbital period."
            )
        elif row["period_days"] > 10:
            explanations.append(
                "Candidate has a moderate orbital period."
            )
        else:
            explanations.append(
                "Candidate has a short orbital period."
            )

        if row["scientific_score"] > 0.80:
            explanations.append(
                "High-priority candidate for follow-up study."
            )
        elif row["scientific_score"] > 0.60:
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
                y=result["global_view"],
                mode="lines",
                name="Global View"
            )
        )

        global_fig.update_layout(
            title="Global Transit View"
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
            title="Local Transit View"
        )
        
        pred_text = html.Div([

            html.Div([

                html.Div([
                    html.H6("Prediction"),
                    html.H4(result["prediction"])
                ], style=CARD_STYLE),

                html.Div([
                    html.H6("Confidence"),
                    html.H4(f"{result['confidence']:.2%}")
                ], style=CARD_STYLE),

                html.Div([
                    html.H6("Scientific Score"),
                    html.H4(f"{result['scientific_score']:.4f}")
                ], style=CARD_STYLE),

                html.Div([
                    html.H6("SNR"),
                    html.H4(f"{result['snr']:.2f}")
                ], style=CARD_STYLE),

            ], style={
                "display": "flex",
                "gap": "20px",
                "justifyContent": "center",
                "flexWrap": "wrap"
            }),

            html.Br(),

            html.Div([

                html.Div([
                    html.H6("Period (days)"),
                    html.H4(f"{result['period_days']:.2f}")
                ], style=CARD_STYLE),

                html.Div([
                    html.H6("Duration (hours)"),
                    html.H4(f"{result['duration_hours']:.2f}")
                ], style=CARD_STYLE),

                html.Div([
                    html.H6("Depth (ppm)"),
                    html.H4(f"{result['depth_ppm']:.1f}")
                ], style=CARD_STYLE),

            ], style={
                "display": "flex",
                "gap": "20px",
                "justifyContent": "center",
                "flexWrap": "wrap"
            }),

            html.Hr(),

            html.H3("Prediction Explanation"),

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

    except Exception as e:

        print("CALLBACK ERROR:", e)

        return (
            html.Div(f"ERROR: {e}"),
            go.Figure(),
            go.Figure()
        )

@app.callback(
    Output(
        "download-csv",
        "data"
    ),

    Input(
        "download-btn",
        "n_clicks"
    ),

    prevent_initial_call=True
)
def download_csv(n_clicks):

    return dcc.send_file(
        "candidate_ranking.csv"
    )

if __name__ == "__main__":
    app.run(debug=True)