from predict import predict_npz


def explain_candidate(path):

    r = predict_npz(path)

    print("\nEXOPLANET CANDIDATE REPORT")
    print("=" * 50)

    print(f"Prediction       : {r['prediction']}")
    print(f"Confidence       : {r['confidence']:.4f}")
    print(f"Scientific Score : {r['scientific_score']:.4f}")

    print()
    print("Transit Features")
    print("-" * 50)

    print(f"Period           : {r['period_days']:.4f} days")
    print(f"Duration         : {r['duration_hours']:.4f} hours")
    print(f"Depth            : {r['depth_ppm']:.2f} ppm")
    print(f"SNR              : {r['snr']:.2f}")

    print()
    print("Interpretation")
    print("-" * 50)

    if r["confidence"] > 0.90:
        print("Model confidence is very high.")
    elif r["confidence"] > 0.75:
        print("Model confidence is moderate.")
    else:
        print("Model confidence is low.")

    if r["snr"] > 20:
        print("Strong signal-to-noise ratio.")
    elif r["snr"] > 10:
        print("Acceptable signal-to-noise ratio.")
    else:
        print("Weak signal-to-noise ratio.")

    if r["depth_ppm"] > 500:
        print("Transit depth is clearly visible.")
    else:
        print("Transit depth is relatively shallow.")

    if r["scientific_score"] > 0.80:
        print("High-priority candidate for follow-up study.")
    elif r["scientific_score"] > 0.60:
        print("Moderately interesting candidate.")
    else:
        print("Low-priority candidate.")

    print("=" * 50)


if __name__ == "__main__":

    explain_candidate(
        "data/processed/4455231_K01332.03.npz"
    )