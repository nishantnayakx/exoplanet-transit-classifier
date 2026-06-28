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

print("=" * 60)
print("Dashboard starting...")
print("Current working directory:", os.getcwd())
print("DATA_DIR:", DATA_DIR)
print("Exists:", os.path.exists(DATA_DIR))

if os.path.exists(DATA_DIR):
    print("Files found:", len(glob.glob(os.path.join(DATA_DIR, "*.npz"))))
    print(glob.glob(os.path.join(DATA_DIR, "*.npz"))[:5])
else:
    print("DATA DIRECTORY NOT FOUND!")

print("=" * 60)

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
    title="Model Performance Metrics"
)

metric_fig.update_layout(
    xaxis_title="Evaluation Metric",
    yaxis_title="Score"
)

prediction_fig = px.pie(
    ranking_df,
    names="prediction",
    title="Distribution of Predicted Classes"
)

confidence_fig = px.histogram(
    ranking_df,
    x="confidence",
    nbins=30,
    title="Prediction Confidence Distribution"
)

confidence_fig.update_layout(
    xaxis_title="Prediction Confidence",
    yaxis_title="Number of Candidates"
)

ranking_fig = px.bar(
    top20_df.head(10),
    x="scientific_score",
    y="file",
    orientation="h",
    color="prediction",
    color_discrete_map={

        "planet_transit": "#2ecc71",
        "false_positive": "#e74c3c"

    },
    title="Top 10 Highest Ranked Planet Transit Candidates",
    custom_data=[
        "prediction",
        "confidence",
        "snr",
        "period_days",
        "duration_hours",
        "depth_ppm"
    ],

)

ranking_fig.update_traces(

    hovertemplate=
        "<b>%{y}</b><br>" +

        "Scientific Score: %{x:.3f}<br>" +

        "Prediction: %{customdata[0]}<br>" +

        "Confidence: %{customdata[1]:.3f}<br>" +

        "SNR: %{customdata[2]:.2f}<br>" +

        "Period: %{customdata[3]:.2f} days<br>" +

        "Duration: %{customdata[4]:.2f} hrs<br>" +

        "Depth: %{customdata[5]:.1f} ppm" +

        "<extra></extra>"

)

ranking_fig.update_layout(
    xaxis_title="Scientific Priority Score",
    yaxis_title="Kepler Candidate ID",
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
    title="Prediction Confidence vs Scientific Score"
)

scatter_fig.update_traces(

    marker=dict(size=10),

    hovertemplate=
        "<b>%{customdata[0]}</b><br>" +
        "Confidence: %{x:.2%}<br>" +
        "Scientific Score: %{y:.2f}<extra></extra>"
)

scatter_fig.update_layout(
    xaxis_title="Prediction Confidence",
    yaxis_title="Scientific Priority Score"
)

scientific_fig = px.bar(
    ranking_df.head(20),
    x="file",
    y="scientific_score",
    title="Highest Scientific Priority Candidates"
)

scientific_fig.update_traces(

    hovertemplate=
        "<b>%{x}</b><br>" +
        "Scientific Score: %{y:.3f}<extra></extra>"
)

scientific_fig.update_layout(
    xaxis_title="Kepler Candidate ID",
    yaxis_title="Scientific Priority Score"
)

def style_figure(fig):

    fig.update_layout(

        template="plotly_white",

        title_x=0.5,

        title_font=dict(
            size=22
        ),

        font=dict(
            family="Arial",
            size=14
        ),

        paper_bgcolor="white",

        plot_bgcolor="white",

        margin=dict(
            l=40,
            r=40,
            t=70,
            b=40
        ),

        height=500,

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        transition_duration=500

    )

    fig.update_xaxes(showgrid=True)

    fig.update_yaxes(showgrid=True)

    return fig



metric_fig = style_figure(metric_fig)

prediction_fig = style_figure(prediction_fig)

confidence_fig = style_figure(confidence_fig)

ranking_fig = style_figure(ranking_fig)

scientific_fig = style_figure(scientific_fig)

scatter_fig = style_figure(scatter_fig)

CARD_STYLE = {
    "padding": "15px",
    "border": "1px solid #ddd",
    "borderRadius": "10px",
    "textAlign": "center",
    "width": "180px",
    "backgroundColor": "#f8f9fa",
    "boxShadow": "0px 2px 4px rgba(0,0,0,0.08)"
}

METRIC_CARD_STYLE = {
    "padding": "20px",
    "border": "1px solid #ddd",
    "borderRadius": "12px",
    "textAlign": "center",
    "minWidth": "250px",
    "maxWidth": "320px",
    "flex": "1 1 250px",
    "backgroundColor": "white",
    "boxShadow": "0px 4px 10px rgba(0,0,0,0.12)"
}

GRAPH_CARD_STYLE = {
    "backgroundColor": "white",
    "padding": "20px",
    "borderRadius": "12px",
    "boxShadow": "0px 4px 10px rgba(0,0,0,0.08)",
    "marginBottom": "35px"
}

SECTION_CARD_STYLE = {
    "backgroundColor": "white",
    "padding": "25px",
    "borderRadius": "12px",
    "boxShadow": "0px 4px 10px rgba(0,0,0,0.08)",
    "marginBottom": "35px",
    "width":"100%",
    "overflow":"hidden",
    "boxSizing": "border-box"
}

EVALUATION_CARD_STYLE = {
    "backgroundColor": "white",
    "padding": "25px",
    "borderRadius": "12px",
    "boxShadow": "0px 4px 10px rgba(0,0,0,0.08)",
    "marginBottom": "35px"
}

SECTION_TITLE_STYLE = {
    "marginBottom": "18px",
    "fontWeight": "600",
    "color": "#2c3e50"
}

app.layout = html.Div([

         html.Div([

            html.Div([

                html.H1(
                    "Exoplanet Transit Classifier",
                    style={
                        "textAlign": "center",
                        "marginBottom": "10px"
                    }
                ),

                html.P(
                    "An interactive dashboard for deep learning-based exoplanet transit detection and scientific candidate ranking using Kepler light curve data.",
                    style={
                        "textAlign": "center",
                        "fontSize": "18px",
                        "color": "gray"
                    }
                ),

                html.Div([

                    html.A(
                        "GitHub Repository",
                        href="https://github.com/nishantnayakx/exoplanet-transit-classifier",
                        target="_blank",
                        style={
                            "padding": "10px 20px",
                            "border": "1px solid #007bff",
                            "borderRadius": "8px",
                            "textDecoration": "none"
                        }
                    ),

                    html.A(
                        "Live Dashboard",
                        href="https://exoplanet-transit-classifier.onrender.com/",
                        target="_blank",
                        style={
                            "padding": "10px 20px",
                            "border": "1px solid #28a745",
                            "borderRadius": "8px",
                            "textDecoration": "none"
                        }
                    ),

                ], style={
                    "display": "flex",
                    "gap": "20px",
                    "justifyContent": "center",
                    "marginTop": "15px",
                    "marginBottom": "30px"
                })

            ]),

            html.Div([

                html.H2("🚀 About This Project"),

                html.P(
                    "This dashboard demonstrates a deep learning pipeline for detecting exoplanet transit signals from NASA Kepler light curve data. "
                    "The model classifies transit candidates, ranks them using a scientific priority score, and provides interactive visualizations "
                    "to assist astronomers in identifying promising exoplanet candidates for follow-up observations.",
                    style={
                        "lineHeight": "1.8",
                        "fontSize": "16px",
                        "textAlign": "justify"
                    }
                )

            ], style=SECTION_CARD_STYLE),


            html.Div([

            html.Div([
                html.H3(f"{len(ranking_df)}"),
                html.P("Kepler Samples")
            ], style={
                **METRIC_CARD_STYLE,
                "backgroundColor": "#eaf4ff"
            }),

            html.Div([
                html.H3(
                    f"{(ranking_df['prediction']=='planet_transit').sum()}"
                ),
                html.P("Predicted Exoplanet Candidates")
            ], style={
                **METRIC_CARD_STYLE,
                "backgroundColor": "#eefbf3"
            }),


            html.Div([
                html.H3(
                    f"{(ranking_df['prediction']=='false_positive').sum()}"
                ),
                html.P("Predicted False Positives")
            ], style={
                **METRIC_CARD_STYLE,
                "backgroundColor": "#fff1f1"
            }),


            html.Div([
                html.H3(f"{metrics.loc[metrics['Metric']=='ROC-AUC','Value'].iloc[0]:.4f}"),
                html.P("Model ROC–AUC")
            ], style={
                **METRIC_CARD_STYLE,
                "backgroundColor": "#f4efff"
            }),

        ], style={
            "display": "flex",
            "flexWrap": "wrap",
            "justifyContent": "center",
            "alignItems": "stretch",
            "gap": "20px",
            "marginBottom": "30px",
            "width": "100%"
        }),

    html.Div([

            html.H2("📊 Model Metrics", style=SECTION_TITLE_STYLE),
            html.P(
                "Summary of the classification performance achieved by the trained neural network on the evaluation dataset.",
                style={
                    "color": "gray",
                    "marginBottom": "15px"
                }
            ),

            dcc.Graph(
                figure=metric_fig
            )

        ], style=GRAPH_CARD_STYLE),


    html.Div([

            html.H2("🪐 Prediction Distribution Across Dataset", style=SECTION_TITLE_STYLE),

            html.P(

                "Distribution of predicted exoplanet transits and false positives across all processed Kepler candidates.",

                style={

                    "color":"gray",

                    "marginBottom":"15px"

                }

            ),

            dcc.Graph(
                figure=prediction_fig
            )

        ], style=GRAPH_CARD_STYLE),


        html.Div([

            html.H2("🎯 Model Confidence Distribution", style=SECTION_TITLE_STYLE),

            html.P(

                "Model confidence scores for every candidate prediction generated by the trained neural network classifier.",

                style={

                    "color":"gray",

                    "marginBottom":"15px"

                }

            ),

            dcc.Graph(
                figure=confidence_fig
            )

        ], style=GRAPH_CARD_STYLE),


        html.Div([

            html.H2("⭐ Highest Scientific Priority Candidates", style=SECTION_TITLE_STYLE),
            html.P(
                "Candidates ranked by the proposed scientific priority scores recommended for follow-up astronomical observations.",
                style={
                    "color":"gray",
                    "marginBottom":"15px"
                }
            ),

            dcc.Graph(
                figure=scientific_fig
            )

        ], style=GRAPH_CARD_STYLE),


        html.Div([

            html.H2("🏆 Top Ranked Candidates", style=SECTION_TITLE_STYLE),
            html.P(
                "Highest-priority exoplanet candidates selected according to the proposed scientific ranking framework used by the trained neural network classifier.",
                style={
                    "color": "gray",
                    "marginBottom": "15px"
                }
            ),

            dcc.Graph(
                figure=ranking_fig
            )

        ], style=GRAPH_CARD_STYLE),


        html.Div([

            html.H2("🔬 Scientific Candidate Analysis", style=SECTION_TITLE_STYLE),
            html.P(
                "Relationship between scientific priority score and important transit characteristics.",
                style={
                    "color":"gray",
                    "marginBottom":"15px"
                }
            ),

            dcc.Graph(
                figure=scatter_fig
            )

        ], style=GRAPH_CARD_STYLE),

        html.Div([

            html.H2("📋 Candidate Rankings", style=SECTION_TITLE_STYLE),

            html.P(
                "Browse, search, sort, filter, and export the highest-ranked Kepler candidates based on their predicted scientific priority.",
                style={
                    "color": "gray",
                    "marginBottom": "15px"
                }
            ),

            html.Button(
                "⬇ Download Ranked Candidates (.csv)",
                id="download-btn",
                style={
                    "backgroundColor":"#007bff",
                    "color":"white",
                    "padding":"12px 22px",
                    "border":"none",
                    "borderRadius":"8px",
                    "cursor":"pointer",
                    "fontWeight":"bold",
                    "marginBottom":"20px"
                }
            ),

            dcc.Download(id="download-csv"),

            html.Div(

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

                        "overflowX":"auto",

                        "borderRadius":"10px",

                        "overflow":"hidden",

                        "width":"100%",

                        "minWidth":"100%"

                    },

                    style_data={

                        "backgroundColor":"white",

                        "border":"1px solid #eeeeee"

                    },

                    style_cell={

                        "textAlign":"left",

                        "padding":"12px",

                        "fontFamily":"Arial",

                        "fontSize":"14px",

                        "whiteSpace":"normal",

                        "height":"auto",

                        "minWidth":"120px",

                        "maxWidth":"250px"

                    },

                    style_header={

                        "backgroundColor":"#2c3e50",

                        "color":"white",

                        "fontWeight":"bold",

                        "textAlign":"center",

                        "fontSize":"15px"

                    },

                    fixed_rows={"headers": True}
                ),

                style={
                    "overflowX":"auto"     
                }

            ),    

        ], style=SECTION_CARD_STYLE),


        html.Div([

            html.H2("🔭 Candidate Explorer", style=SECTION_TITLE_STYLE),
            html.P(
                "Interactively inspect an individual candidate, visualize its transit curves, and review the model's prediction with scientific explanations.",
                style={
                    "color":"gray",
                    "marginBottom":"15px"
                }
            ),

            html.P(
                "Select a Kepler candidate to inspect the model prediction, confidence, scientific score, and transit light curves.",
                style={
                    "color": "gray",
                    "marginBottom": "15px"
                }
            ),
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

            dcc.Loading(

                type="circle",

                children=[

                    html.Div(
                        id="prediction-output",
                        style={
                            "marginTop": "15px",
                            "marginBottom": "10px"
                        }
                    ),

                    html.Br(),

                    html.Div([

                        html.Div([

                            html.H3(
                                "🌌 Global Transit Curve",
                                style={
                                    "color": "#2c3e50"
                                }
                            ),

                            html.P(
                                "Entire folded Kepler light curve used by the neural network.",
                                style={
                                    "color": "gray",
                                    "marginBottom": "10px"
                                }
                            ),

                            dcc.Graph(
                                id="global-view-graph"
                            ),

                        ], style={
                            "flex": "1",
                            "minWidth": "450px"
                        }),

                        html.Div([

                            html.H3(
                                "🔍 Zoomed Transit Curve",
                                style={
                                    "color": "#2c3e50"
                                }
                            ),

                            html.P(
                                "Detailed view centered on the transit event.",
                                style={
                                    "color": "gray",
                                    "marginBottom": "10px"
                                }
                            ),

                            dcc.Graph(
                                id="local-view-graph"
                            ),

                        ], style={
                            "flex": "1",
                            "minWidth": "450px"
                        }),

                    ], style={

                        "display": "flex",

                        "gap": "25px",

                        "flexWrap": "wrap",

                        "alignItems": "flex-start"

                    }),

                ]

            ),

        ], style=SECTION_CARD_STYLE),   


        html.Div([

            html.H2("📈 Model Performance Evaluation", style=SECTION_TITLE_STYLE),

            html.P(
                "Visual evaluation of the classifier using confusion matrix, ROC curve, and Precision–Recall curve.",
                style={
                    "color":"gray",
                    "marginBottom":"15px"
                }
            ),

            html.H4("📋 Confusion Matrix"),
            html.P(
                "Comparison between ground-truth labels and model predictions on the evaluation dataset.",
                style={
                    "color":"gray",
                    "marginBottom":"15px"
                }
            ),

            html.Div(

                html.Img(
                    src="/assets/confusion_matrix.png",
                    style={
                        "width": "100%",
                        "maxWidth": "700px",
                        "display": "block",
                        "margin": "0 auto"
                    }
                ),

                style={
                    "overflow": "hidden"
                }

            ),

            html.H4("📈 Receiver Operating Characteristic (ROC)"),
            html.P(
                "ROC curve illustrating the classifier's ability to distinguish between exoplanet transit and false positive classes.",
                style={
                    "color":"gray",
                    "marginBottom":"15px"
                }
            ),

            html.Div(

                html.Img(
                    src="/assets/roc_curve.png",
                    style={
                        "width": "100%",
                        "maxWidth": "700px",
                        "display": "block",
                        "margin": "0 auto"
                    }
                ),

                style={
                    "overflow": "hidden"
                }

            ),

            html.H4("📉 Precision–Recall Curve"),
            html.P(
                "Precision–Recall trade-off across different prediction thresholds.",
                style={
                    "color":"gray",
                    "marginBottom":"15px"
                }
            ),

            html.Div(

                html.Img(
                    src="/assets/pr_curve.png",
                    style={
                        "width": "100%",
                        "maxWidth": "700px",
                        "display": "block",
                        "margin": "0 auto"
                    }
                ),

                style={
                    "overflow": "hidden"
                }

            ),

        ], style=EVALUATION_CARD_STYLE),


        html.Div([

            html.H2("🏗️ System Architecture", style=SECTION_TITLE_STYLE),
            html.P(
                "Overview of the complete deep learning pipeline from Kepler light curve preprocessing to candidate ranking and dashboard visualization.",
                style={
                    "color":"gray",
                    "marginBottom":"15px"
                }
            ),
            html.Div(

                html.Img(
                    src="/assets/architecture_diagram.png",
                    style={
                        "width": "100%",
                        "maxWidth": "1000px",
                        "display": "block",
                        "margin": "0 auto"
                    }
                ),

                style={
                    "overflow": "hidden"
                }

            ),

        ], style=SECTION_CARD_STYLE),

       

        html.Footer([

            html.Hr(),

            html.P(
                "Developed by Nishant Nayak",
                style={"fontWeight": "bold"}
            ),

            html.P(
                "Deep Learning • PyTorch • Dash • Plotly • NASA Kepler Dataset"
            ),

            html.Div([

                html.A(
                    "GitHub Repository",
                    href="https://github.com/nishantnayakx/exoplanet-transit-classifier",
                    target="_blank"
                ),

                html.Span(" | "),

                html.A(
                    "Live Dashboard",
                    href="https://exoplanet-transit-classifier.onrender.com/",
                    target="_blank"
                )

            ])

        ],

        style={

            "textAlign":"center",

            "padding":"30px",

            "color":"gray"

        })

    ], style={
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "20px"
    }),    

], style={

    "backgroundColor": "#f4f6f9",

    "padding": "30px"

})


@app.callback(
    [
        Output("prediction-output", "children"),
        Output("global-view-graph", "figure"),
        Output("local-view-graph", "figure")
    ],
    Input("candidate-dropdown", "value")
)
def update_candidate(path):

    print("=" * 60)
    print("STEP 2")
    print(path)

    result = predict_npz(path)

    print("predict_npz finished")

    return (

        html.Div([

            html.H2("Prediction Test"),

            html.P(result["prediction"]),

            html.P(result["confidence"])

        ]),

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