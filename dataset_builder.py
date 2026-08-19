import os
import json
import subprocess
import sys
import pandas as pd


# ============================================================
#                 PULSEGUARD DATASET BUILDER
# ============================================================

REAL_FOLDER = "videos/real_clips"
FAKE_FOLDER = "videos/fake_clips"

OUTPUT_FILE = "dataset.csv"

FEATURE_FILE = "features.json"


# ============================================================
# GET VIDEO FILES
# ============================================================

def get_videos(folder):

    if not os.path.exists(folder):

        print(
            f"ERROR: Folder does not exist: {folder}"
        )

        return []

    valid_extensions = (

        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    )

    videos = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            valid_extensions
        ):

            videos.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    return sorted(videos)


# ============================================================
# ANALYZE ONE VIDEO
# ============================================================

def analyze_video(
    video_path,
    label
):

    print()
    print("=" * 60)

    print(
        "Analyzing:",
        video_path
    )

    print(
        "Label:",
        "REAL" if label == 0 else "FAKE"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Run main.py
    # --------------------------------------------------------

    if os.path.exists(FEATURE_FILE):
        os.remove(FEATURE_FILE)

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            video_path
        ],
        capture_output=True,
        text=True
    )


    # Print program output

    print(
        result.stdout
    )


    if result.returncode != 0:

        print(
            "ERROR while processing:",
            video_path
        )

        print(
            result.stderr
        )

        return None


    # --------------------------------------------------------
    # Read generated features
    # --------------------------------------------------------

    if not os.path.exists(
        FEATURE_FILE
    ):

        print(
            "ERROR: features.json was not created."
        )

        return None


    with open(
        FEATURE_FILE,
        "r"
    ) as f:

        features = json.load(f)


    # Add label

    features["label"] = label

    features["video"] = video_path


    return features


# ============================================================
# MAIN DATASET BUILDER
# ============================================================

real_videos = get_videos(
    REAL_FOLDER
)

fake_videos = get_videos(
    FAKE_FOLDER
)


print()
print("=" * 60)
print("             PULSEGUARD DATASET BUILDER")
print("=" * 60)

print()
print(
    "Real videos:",
    len(real_videos)
)

print(
    "Fake videos:",
    len(fake_videos)
)


# ============================================================
# PROCESS ALL VIDEOS
# ============================================================

all_features = []


# ------------------------------------------------------------
# REAL VIDEOS
# ------------------------------------------------------------

for video in real_videos:

    features = analyze_video(
        video,
        0
    )

    if features is not None:

        all_features.append(
            features
        )


# ------------------------------------------------------------
# FAKE VIDEOS
# ------------------------------------------------------------

for video in fake_videos:

    features = analyze_video(
        video,
        1
    )

    if features is not None:

        all_features.append(
            features
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

if len(all_features) == 0:

    print()
    print(
        "ERROR: No videos were successfully processed."
    )

    sys.exit()


df = pd.DataFrame(
    all_features
)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("             DATASET COMPLETE")
print("=" * 60)

print()

print(
    "Total videos:",
    len(df)
)

print(
    "Real videos:",
    int(
        (df["label"] == 0).sum()
    )
)

print(
    "Fake videos:",
    int(
        (df["label"] == 1).sum()
    )
)

print()

print(
    "Dataset shape:",
    df.shape
)

print()

print(
    "Saved:",
    OUTPUT_FILE
)

print()

print(
    df
)

print()
print("=" * 60)