import pandas as pd

html = f"""
<html>
<head>
<title>Exoplanet Transit Classifier Report</title>

<style>
body {{
    font-family: Arial;
    margin: 40px;
}}

img {{
    max-width: 800px;
}}

table {{
    border-collapse: collapse;
}}

th, td {{
    border: 1px solid black;
    padding: 8px;
}}
</style>

</head>

<body>

<h1>Exoplanet Transit Classifier</h1>

<h2>Project Summary</h2>

<p>
This project uses a deep learning based dual-view CNN
to identify potential exoplanet transit signals from
Kepler light curve observations.
</p>

<h2>Model Performance</h2>

<ul>
<li>Validation Accuracy: 78.7%</li>
<li>ROC AUC: 0.8424</li>
<li>Planet Precision: 0.82</li>
<li>Planet Recall: 0.87</li>
<li>Planet F1: 0.84</li>
</ul>

<h2>Confusion Matrix</h2>

<img src="../assets/confusion_matrix.png">

<h2>ROC Curve</h2>

<img src="../assets/roc_curve.png">

<h2>Precision Recall Curve</h2>

<img src="../assets/pr_curve.png">

<h2>Top Candidates</h2>

{pd.read_csv("candidate_ranking.csv").head(20).to_html(index=False)}

<h2>Methodology</h2>

<ul>
<li>Kepler Candidate Data</li>
<li>Global View Representation</li>
<li>Local Transit View Representation</li>
<li>Dual Branch CNN Architecture</li>
<li>Probability Ranking System</li>
</ul>

<h2>Future Improvements</h2>

<ul>
<li>Explainable AI visualizations</li>
<li>Attention mechanisms</li>
<li>Multi-mission support (Kepler + TESS)</li>
<li>Real-time deployment API</li>
</ul>

</body>
</html>
"""

with open(
    "reports/exoplanet_report.html",
    "w",
    encoding="utf-8"
) as f:
    f.write(html)

print("Report generated")