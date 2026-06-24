import pandas as pd
import dash
from dash import html, dcc
import plotly.express as px

app = dash.Dash(__name__)

ranking_df = pd.read_csv("candidate_ranking.csv")

top20_df = ranking_df.head(20)

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

app.layout = html.Div([


html.H1("Exoplanet Transit Classifier"),

html.Hr(),

html.H2("Project Summary"),

html.P(
    """
    Deep learning pipeline for identifying
    exoplanet transit candidates from Kepler
    light curves using a dual-view CNN.
    """
),

html.Hr(),

html.H2("Dataset Overview"),

html.Ul([
    html.Li("Total Samples: 943"),
    html.Li("Planet Transits: 593"),
    html.Li("False Positives: 350")
]),

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

html.H2("Top Ranked Candidates"),

html.Table(

    [

        html.Tr([
            html.Th("Rank"),
            html.Th("File"),
            html.Th("Prediction"),
            html.Th("Confidence"),
            html.Th("Scientific Score")
        ])

    ] +

    [

        html.Tr([

            html.Td(row["rank"]),
            html.Td(row["file"]),
            html.Td(row["prediction"]),
            html.Td(round(row["confidence"], 4)),
            html.Td(round(row["scientific_score"], 4))

        ])

        for _, row in top20_df.iterrows()

    ]

),

html.Hr(),

html.H2("ROC Curve"),

html.Img(
    src="/assets/roc_curve.png",
    style={"width": "900px"}
),

html.Hr(),

html.H2("Precision Recall Curve"),

html.Img(
    src="/assets/pr_curve.png",
    style={"width": "900px"}
),

html.Hr(),

html.H2("Confusion Matrix"),

html.Img(
    src="/assets/confusion_matrix.png",
    style={"width": "900px"}
),

html.Hr(),

html.H2("Architecture"),

html.Img(
    src="/assets/architecture_diagram.png",
    style={"width": "900px"}
)


])

server = app.server

if __name__ == "__main__":
    app.run(debug=False)
