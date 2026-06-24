from predict import predict_npz


def generate_explanation(result):

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

    return explanations


def explain_candidate(path):

    r = predict_npz(path)

    explanations = generate_explanation(r)

    print("\nEXOPLANET CANDIDATE REPORT")
    print("=" * 60)

    print(f"Prediction       : {r['prediction']}")
    print(f"Confidence       : {r['confidence']:.4f}")
    print(f"Scientific Score : {r['scientific_score']:.4f}")

    print()
    print("Transit Features")
    print("-" * 60)

    print(f"Period           : {r['period_days']:.4f} days")
    print(f"Duration         : {r['duration_hours']:.4f} hours")
    print(f"Depth            : {r['depth_ppm']:.2f} ppm")
    print(f"SNR              : {r['snr']:.2f}")

    print()
    print("Interpretation")
    print("-" * 60)

    for item in explanations:
        print(f"- {item}")

    print("=" * 60)

    return explanations


if __name__ == "__main__":

    explain_candidate(
        "data/processed/4455231_K01332.03.npz"
    )