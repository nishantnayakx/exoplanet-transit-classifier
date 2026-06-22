import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

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

fig = px.bar(
    metrics,
    x="Metric",
    y="Value",
    title="Model Performance"
)

app.layout = html.Div([
    html.H1("Exoplanet Transit Classifier Dashboard"),

    html.H2("Dataset Overview"),

    html.Ul([
        html.Li("Total Samples: 943"),
        html.Li("Planet Transits: 593"),
        html.Li("False Positives: 350")
    ]),

    html.H2("Model Metrics"),

    dcc.Graph(figure=fig),

    html.H2("Confusion Matrix"),

    html.Img(
        src="/assets/confusion_matrix.png",
        style={"width": "600px"}
    )
])

if __name__ == "__main__":
    app.run(debug=True)