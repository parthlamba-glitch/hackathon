import json
import joblib
import pandas as pd


# ============================================================
#                    CONFIGURATION
# ============================================================
MODEL_FILE = "pulseguard_model.pkl"
FEATURE_FILE = "features.json"


FEATURES = [

    "face_detection_rate",

    "median_bpm",

    "mean_peak_strength",

    "left_region_bpm",

    "right_region_bpm",

    "regional_bpm_std",

    "temporal_variability",

    "forehead_left_corr",

    "forehead_right_corr",

    "left_right_corr",

    "regional_coherence"
]


# ============================================================
#                    LOAD MODEL
# ============================================================

print("Loading PulseGuard model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded.")


# ============================================================
#                    LOAD FEATURES
# ============================================================

print("Loading extracted features...")

with open(
    FEATURE_FILE,
    "r"
) as f:

    features = json.load(f)

print("Features loaded.")


# ============================================================
#                    CREATE INPUT
# ============================================================

input_data = {

    feature: features.get(
        feature,
        None
    )

    for feature in FEATURES
}


X = pd.DataFrame(
    [input_data]
)


# ============================================================
#                    PREDICTION
# ============================================================

prediction = model.predict(
    X
)[0]

probabilities = model.predict_proba(
    X
)[0]


real_probability = float(
    probabilities[0]
)

fake_probability = float(
    probabilities[1]
)


# ============================================================
#                    VERDICT
# ============================================================

if fake_probability >= 0.5:

    verdict = "LIKELY FAKE"

    score = fake_probability

else:

    verdict = "LIKELY REAL"

    score = real_probability


# ============================================================
#                    RESULT
# ============================================================

result = {

    "verdict": verdict,

    "score": round(
        float(score),
        4
    ),

    "real_probability": round(
        real_probability,
        4
    ),

    "fake_probability": round(
        fake_probability,
        4
    ),

    "bpm": features.get(
        "median_bpm"
    ),

    "face_detection_rate":
        features.get(
            "face_detection_rate"
        ),

    "regional_coherence":
        features.get(
            "regional_coherence"
        ),

    "temporal_variability":
        features.get(
            "temporal_variability"
        )
}


# ============================================================
#                    OUTPUT
# ============================================================

print()
print("=" * 60)
print("                 PULSEGARD PREDICTION")
print("=" * 60)

print()

print(
    "Verdict:",
    result["verdict"]
)

print(
    "Model score:",
    f"{result['score'] * 100:.2f}%"
)

print(
    "Real probability:",
    f"{result['real_probability'] * 100:.2f}%"
)

print(
    "Fake probability:",
    f"{result['fake_probability'] * 100:.2f}%"
)

print()

print(
    "BPM:",
    result["bpm"]
)

print(
    "Face detection:",
    result["face_detection_rate"]
)

print(
    "Regional coherence:",
    result["regional_coherence"]
)

print(
    "Temporal variability:",
    result["temporal_variability"]
)

print()

print(
    json.dumps(
        result,
        indent=4
    )
)