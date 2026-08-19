import json
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.signal import butter, sosfiltfilt


# ============================================================
#                    PULSEGUARD
#          rPPG / PHYSIOLOGICAL SIGNAL PIPELINE
# ============================================================

print()
print("=" * 60)
print("                 PULSEGUARD STARTING")
print("=" * 60)
print()


# ============================================================
# 1. CONFIGURATION
# ============================================================

VIDEO_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "test_video.mp4"
)

MODEL_PATH = "face_landmarker.task"

# ------------------------------------------------------------
# Plot control
# ------------------------------------------------------------
# True  -> display debugging plots
# False -> close plots automatically
#
# IMPORTANT:
# Keep this False when processing many videos.
# ------------------------------------------------------------

SHOW_PLOTS = False


# ------------------------------------------------------------
# Heart-rate frequency range
# ------------------------------------------------------------

LOW_HZ = 0.7
HIGH_HZ = 4.0


# ------------------------------------------------------------
# POS window
# ------------------------------------------------------------

POS_WINDOW_SECONDS = 1.6


# ------------------------------------------------------------
# Windowed BPM analysis
# ------------------------------------------------------------

BPM_WINDOW_SECONDS = 10
BPM_STEP_SECONDS = 5


# ------------------------------------------------------------
# FFT size
# ------------------------------------------------------------

FFT_SIZE = 4096


# ============================================================
# 2. VIDEO SETUP
# ============================================================

video_path = VIDEO_PATH

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    print("ERROR: Could not open video.")
    sys.exit(1)


fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


if fps <= 0:

    print("ERROR: Invalid FPS.")

    cap.release()

    sys.exit(1)


duration_seconds = (
    total_frames / fps
)


print(
    "Video opened successfully"
)

print(
    "FPS:",
    round(fps, 2)
)

print(
    "Total frames:",
    total_frames
)

print(
    "Resolution:",
    width,
    "x",
    height
)

print(
    "Duration:",
    round(
        duration_seconds,
        2
    ),
    "seconds"
)

print()


# ============================================================
# 3. MEDIAPIPE FACE LANDMARKER SETUP
# ============================================================

print(
    "Initializing MediaPipe..."
)


BaseOptions = (
    mp.tasks.BaseOptions
)

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


landmarker = (
    FaceLandmarker.create_from_options(
        options
    )
)


print(
    "MediaPipe initialized."
)

print()


# ============================================================
# 4. STORAGE
# ============================================================

forehead_rgb = []

left_cheek_rgb = []

right_cheek_rgb = []

# Actual timestamps corresponding to
# successfully detected faces.

sample_times = []

processed_frames = 0

face_detected_frames = 0


# ============================================================
# 5. PROCESS VIDEO FRAME-BY-FRAME
# ============================================================

print(
    "Processing video..."
)

print()


while True:

    ret, frame = cap.read()

    if not ret:
        break


    processed_frames += 1


    # --------------------------------------------------------
    # OpenCV gives BGR.
    # MediaPipe expects RGB.
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(

        image_format=(
            mp.ImageFormat.SRGB
        ),

        data=rgb_frame
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    current_time = (
        (processed_frames - 1)
        / fps
    )

    timestamp_ms = int(
        current_time * 1000
    )


    # --------------------------------------------------------
    # Face detection
    # --------------------------------------------------------

    result = (
        landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )
    )


    # --------------------------------------------------------
    # No face found
    # --------------------------------------------------------

    if not result.face_landmarks:
        continue


    face_detected_frames += 1


    # First detected face

    landmarks = (
        result.face_landmarks[0]
    )


    # ========================================================
    # 6. FOREHEAD ROI
    # ========================================================

    forehead_points = [

        landmarks[10],

        landmarks[338],

        landmarks[297],

        landmarks[67],

        landmarks[109],

        landmarks[103]
    ]


    x_coords = [

        int(point.x * width)

        for point in forehead_points
    ]


    y_coords = [

        int(point.y * height)

        for point in forehead_points
    ]


    fx_min = max(
        min(x_coords),
        0
    )

    fx_max = min(
        max(x_coords),
        width
    )

    fy_min = max(
        min(y_coords),
        0
    )

    fy_max = min(
        max(y_coords),
        height
    )


    forehead_roi = frame[
        fy_min:fy_max,
        fx_min:fx_max
    ]


    # ========================================================
    # 7. LEFT CHEEK ROI
    # ========================================================

    left_cheek_points = [

        landmarks[50],

        landmarks[101],

        landmarks[205],

        landmarks[187]
    ]


    x_coords = [

        int(point.x * width)

        for point in left_cheek_points
    ]


    y_coords = [

        int(point.y * height)

        for point in left_cheek_points
    ]


    lx_min = max(
        min(x_coords),
        0
    )

    lx_max = min(
        max(x_coords),
        width
    )

    ly_min = max(
        min(y_coords),
        0
    )

    ly_max = min(
        max(y_coords),
        height
    )


    left_cheek_roi = frame[
        ly_min:ly_max,
        lx_min:lx_max
    ]


    # ========================================================
    # 8. RIGHT CHEEK ROI
    # ========================================================

    right_cheek_points = [

        landmarks[280],

        landmarks[330],

        landmarks[425],

        landmarks[411]
    ]


    x_coords = [

        int(point.x * width)

        for point in right_cheek_points
    ]


    y_coords = [

        int(point.y * height)

        for point in right_cheek_points
    ]


    rx_min = max(
        min(x_coords),
        0
    )

    rx_max = min(
        max(x_coords),
        width
    )

    ry_min = max(
        min(y_coords),
        0
    )

    ry_max = min(
        max(y_coords),
        height
    )


    right_cheek_roi = frame[
        ry_min:ry_max,
        rx_min:rx_max
    ]


    # ========================================================
    # 9. CHECK ROI VALIDITY
    # ========================================================

    if (

        forehead_roi.size == 0

        or left_cheek_roi.size == 0

        or right_cheek_roi.size == 0

    ):

        continue


    # ========================================================
    # 10. AVERAGE RGB VALUES
    # ========================================================

    forehead_mean = (
        forehead_roi.mean(
            axis=(0, 1)
        )
    )


    left_cheek_mean = (
        left_cheek_roi.mean(
            axis=(0, 1)
        )
    )


    right_cheek_mean = (
        right_cheek_roi.mean(
            axis=(0, 1)
        )
    )


    # --------------------------------------------------------
    # OpenCV:
    #
    # B G R
    #
    # Convert to:
    #
    # R G B
    # --------------------------------------------------------

    forehead_rgb.append(
        forehead_mean[::-1]
    )

    left_cheek_rgb.append(
        left_cheek_mean[::-1]
    )

    right_cheek_rgb.append(
        right_cheek_mean[::-1]
    )


    # Save timestamp

    sample_times.append(
        current_time
    )


# ============================================================
# 11. CLEAN UP VIDEO / MEDIAPIPE
# ============================================================

cap.release()

landmarker.close()


# ============================================================
# 12. BASIC VIDEO RESULTS
# ============================================================

print()
print("=" * 60)
print("                    VIDEO RESULTS")
print("=" * 60)

print(
    "Processed frames:",
    processed_frames
)

print(
    "Face detected frames:",
    face_detected_frames
)


if processed_frames > 0:

    detection_rate = (

        face_detected_frames
        / processed_frames

    ) * 100

else:

    detection_rate = 0


print(
    "Face detection rate:",
    round(
        detection_rate,
        2
    ),
    "%"
)


# ============================================================
# 13. CONVERT TO NUMPY ARRAYS
# ============================================================

forehead_rgb = np.array(
    forehead_rgb,
    dtype=float
)

left_cheek_rgb = np.array(
    left_cheek_rgb,
    dtype=float
)

right_cheek_rgb = np.array(
    right_cheek_rgb,
    dtype=float
)

sample_times = np.array(
    sample_times,
    dtype=float
)


print()

print(
    "Original forehead RGB shape:",
    forehead_rgb.shape
)

print(
    "Original left cheek RGB shape:",
    left_cheek_rgb.shape
)

print(
    "Original right cheek RGB shape:",
    right_cheek_rgb.shape
)


# ============================================================
# 14. SAFETY CHECK
# ============================================================

if len(sample_times) < 100:

    print()

    print(
        "ERROR: Not enough valid face samples."
    )

    sys.exit(1)


# ============================================================
# 15. INTERPOLATE MISSING FRAMES
# ============================================================

def interpolate_rgb(
    rgb_signal,
    sample_times,
    fps,
    total_duration
):
    """
    Reconstruct a uniformly sampled RGB signal.

    If some frames failed face detection,
    interpolation estimates their values from
    nearby valid samples.
    """

    regular_times = np.arange(

        0,

        total_duration,

        1 / fps
    )


    interpolated = np.zeros(

        (
            len(regular_times),
            3
        )
    )


    for channel in range(3):

        interpolated[:, channel] = (
            np.interp(

                regular_times,

                sample_times,

                rgb_signal[:, channel]
            )
        )


    return (
        interpolated,
        regular_times
    )


print()

print(
    "Interpolating missing samples..."
)


forehead_rgb, regular_times = (
    interpolate_rgb(

        forehead_rgb,

        sample_times,

        fps,

        duration_seconds
    )
)


left_cheek_rgb, _ = (
    interpolate_rgb(

        left_cheek_rgb,

        sample_times,

        fps,

        duration_seconds
    )
)


right_cheek_rgb, _ = (
    interpolate_rgb(

        right_cheek_rgb,

        sample_times,

        fps,

        duration_seconds
    )
)


print(
    "Interpolated forehead RGB shape:",
    forehead_rgb.shape
)


# ============================================================
# 16. RGB NORMALIZATION
# ============================================================

def normalize_rgb(
    rgb_signal
):
    """
    Normalize each RGB channel independently.

    Output approximately has:
        mean = 0
        standard deviation = 1
    """

    mean = np.mean(
        rgb_signal,
        axis=0
    )

    std = np.std(
        rgb_signal,
        axis=0
    )


    # Prevent division by zero

    std[
        std < 1e-8
    ] = 1


    normalized = (

        rgb_signal - mean

    ) / std


    return normalized


print()

print(
    "Normalizing RGB signals..."
)


forehead_normalized = (
    normalize_rgb(
        forehead_rgb
    )
)

left_cheek_normalized = (
    normalize_rgb(
        left_cheek_rgb
    )
)

right_cheek_normalized = (
    normalize_rgb(
        right_cheek_rgb
    )
)


# ============================================================
# 17. RGB NORMALIZATION CHECK
# ============================================================

print()

print(
    "RGB normalization check:"
)

print(
    "Forehead means:",
    np.round(
        np.mean(
            forehead_normalized,
            axis=0
        ),
        4
    )
)

print(
    "Forehead std:",
    np.round(
        np.std(
            forehead_normalized,
            axis=0
        ),
        4
    )
)


# ============================================================
# 18. PLOT NORMALIZED RGB
# ============================================================

time = regular_times

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time,
    forehead_normalized[:, 0],
    label="Red"
)

plt.plot(
    time,
    forehead_normalized[:, 1],
    label="Green"
)

plt.plot(
    time,
    forehead_normalized[:, 2],
    label="Blue"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Normalized intensity"
)

plt.title(
    "PulseGuard - Forehead RGB Signal"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


if SHOW_PLOTS:
    plt.show()
else:
    plt.close()


# ============================================================
# 19. POS rPPG EXTRACTION
# ============================================================

def extract_pos_signal(
    rgb_signal,
    fps,
    window_seconds=1.6
):
    """
    Extract rPPG using the POS method.

    Input:
        RGB signal with shape:
        (frames, 3)

    Columns:
        0 = Red
        1 = Green
        2 = Blue
    """

    number_of_frames = (
        rgb_signal.shape[0]
    )

    window_length = int(
        round(
            window_seconds * fps
        )
    )


    if number_of_frames < window_length:

        raise ValueError(
            "Video is too short for POS."
        )


    pulse_signal = np.zeros(
        number_of_frames
    )

    contribution_count = np.zeros(
        number_of_frames
    )


    # --------------------------------------------------------
    # Sliding windows
    # --------------------------------------------------------

    for start in range(

        0,

        number_of_frames
        - window_length
        + 1

    ):

        end = (
            start
            + window_length
        )


        window = (
            rgb_signal[
                start:end
            ].copy()
        )


        # ----------------------------------------------------
        # Temporal normalization
        # ----------------------------------------------------

        mean_rgb = np.mean(

            window,

            axis=0
        )


        mean_rgb[
            mean_rgb < 1e-8
        ] = 1


        Cn = (
            window
            / mean_rgb
        )


        R = Cn[:, 0]

        G = Cn[:, 1]

        B = Cn[:, 2]


        # ----------------------------------------------------
        # POS projections
        # ----------------------------------------------------

        S1 = (
            G - B
        )

        S2 = (
            -2 * R
            + G
            + B
        )


        # ----------------------------------------------------
        # Adaptive weighting
        # ----------------------------------------------------

        std_s1 = np.std(
            S1
        )

        std_s2 = np.std(
            S2
        )


        if std_s2 < 1e-8:
            continue


        alpha = (
            std_s1
            / std_s2
        )


        # ----------------------------------------------------
        # Combine projections
        # ----------------------------------------------------

        H = (
            S1
            + alpha * S2
        )


        # Remove DC component

        H -= np.mean(H)


        # Normalize

        H_std = np.std(H)


        if H_std < 1e-8:
            continue


        H /= H_std


        # ----------------------------------------------------
        # Overlap-add
        # ----------------------------------------------------

        pulse_signal[
            start:end
        ] += H

        contribution_count[
            start:end
        ] += 1


    # ========================================================
    # Average overlapping windows
    # ========================================================

    valid = (
        contribution_count > 0
    )


    pulse_signal[valid] /= (
        contribution_count[valid]
    )


    # ========================================================
    # Final normalization
    # ========================================================

    pulse_signal -= np.mean(
        pulse_signal
    )


    signal_std = np.std(
        pulse_signal
    )


    if signal_std > 1e-8:

        pulse_signal /= signal_std


    return pulse_signal


print()

print(
    "Extracting POS rPPG signals..."
)


forehead_pulse = (
    extract_pos_signal(

        forehead_rgb,

        fps,

        POS_WINDOW_SECONDS
    )
)

left_cheek_pulse = (
    extract_pos_signal(

        left_cheek_rgb,

        fps,

        POS_WINDOW_SECONDS
    )
)

right_cheek_pulse = (
    extract_pos_signal(

        right_cheek_rgb,

        fps,

        POS_WINDOW_SECONDS
    )
)


# ============================================================
# 20. PLOT RAW POS SIGNALS
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    time,
    forehead_pulse,
    label="Forehead"
)

plt.plot(
    time,
    left_cheek_pulse,
    label="Left Cheek"
)

plt.plot(
    time,
    right_cheek_pulse,
    label="Right Cheek"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "rPPG signal"
)

plt.title(
    "PulseGuard - POS Extracted rPPG Signals"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


if SHOW_PLOTS:
    plt.show()
else:
    plt.close()


# ============================================================
# 21. BANDPASS FILTER
# ============================================================

def bandpass_filter(
    signal,
    fps,
    low_hz=0.7,
    high_hz=4.0,
    order=4
):
    """
    Keep only frequencies corresponding
    to plausible human heart rates.
    """

    nyquist = (
        fps / 2
    )


    low = (
        low_hz
        / nyquist
    )

    high = (
        high_hz
        / nyquist
    )


    if high >= 1:
        high = 0.99


    if low <= 0:
        low = 0.001


    sos = butter(

        order,

        [
            low,
            high
        ],

        btype="bandpass",

        output="sos"
    )


    filtered = (
        sosfiltfilt(
            sos,
            signal
        )
    )


    return filtered


print()

print(
    "Applying bandpass filter..."
)


forehead_filtered = (
    bandpass_filter(

        forehead_pulse,

        fps,

        LOW_HZ,

        HIGH_HZ
    )
)

left_cheek_filtered = (
    bandpass_filter(

        left_cheek_pulse,

        fps,

        LOW_HZ,

        HIGH_HZ
    )
)

right_cheek_filtered = (
    bandpass_filter(

        right_cheek_pulse,

        fps,

        LOW_HZ,

        HIGH_HZ
    )
)


# ============================================================
# 22. PLOT FILTERED SIGNAL
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    time,
    forehead_filtered,
    label="Forehead"
)

plt.plot(
    time,
    left_cheek_filtered,
    label="Left Cheek"
)

plt.plot(
    time,
    right_cheek_filtered,
    label="Right Cheek"
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Filtered rPPG"
)

plt.title(
    "PulseGuard - Filtered rPPG Signal"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


if SHOW_PLOTS:
    plt.show()
else:
    plt.close()


# ============================================================
# 23. CROSS-REGION PHYSIOLOGICAL FEATURES
# ============================================================

def safe_correlation(a, b):

    if len(a) != len(b):
        return np.nan

    if np.std(a) < 1e-8:
        return np.nan

    if np.std(b) < 1e-8:
        return np.nan


    correlation = np.corrcoef(
        a,
        b
    )[0, 1]


    if np.isnan(correlation):
        return np.nan


    return float(correlation)


# ------------------------------------------------------------
# Calculate correlation between facial regions
# ------------------------------------------------------------

forehead_left_corr = safe_correlation(
    forehead_filtered,
    left_cheek_filtered
)

forehead_right_corr = safe_correlation(
    forehead_filtered,
    right_cheek_filtered
)

left_right_corr = safe_correlation(
    left_cheek_filtered,
    right_cheek_filtered
)


# ------------------------------------------------------------
# Overall regional coherence
# ------------------------------------------------------------

correlations = np.array([

    forehead_left_corr,

    forehead_right_corr,

    left_right_corr

], dtype=float)


valid_correlations = correlations[
    ~np.isnan(correlations)
]


if len(valid_correlations) > 0:

    regional_coherence = float(
        np.mean(
            np.abs(
                valid_correlations
            )
        )
    )

else:

    regional_coherence = np.nan


print()
print("=" * 60)
print("          CROSS-REGION PHYSIOLOGY")
print("=" * 60)

print()

print(
    "Forehead <-> Left cheek:",
    (
        round(
            forehead_left_corr,
            3
        )
        if not np.isnan(
            forehead_left_corr
        )
        else "NaN"
    )
)

print(
    "Forehead <-> Right cheek:",
    (
        round(
            forehead_right_corr,
            3
        )
        if not np.isnan(
            forehead_right_corr
        )
        else "NaN"
    )
)

print(
    "Left cheek <-> Right cheek:",
    (
        round(
            left_right_corr,
            3
        )
        if not np.isnan(
            left_right_corr
        )
        else "NaN"
    )
)

print(
    "Regional coherence:",
    (
        round(
            regional_coherence,
            3
        )
        if not np.isnan(
            regional_coherence
        )
        else "NaN"
    )
)


# ============================================================
# 24. WINDOWED BPM ANALYSIS
# ============================================================

def analyze_bpm_windows(
    signal,
    fps,
    window_seconds=10,
    step_seconds=5,
    low_hz=0.7,
    high_hz=4.0,
    fft_size=4096
):
    """
    Analyze the signal in overlapping windows.

    Each window produces:
        BPM
        dominant frequency
        spectral peak strength
    """

    window_size = int(
        window_seconds * fps
    )

    step_size = int(
        step_seconds * fps
    )


    results = []


    # --------------------------------------------------------
    # Move through the signal
    # --------------------------------------------------------

    for start in range(

        0,

        len(signal)
        - window_size
        + 1,

        step_size

    ):

        end = (
            start
            + window_size
        )


        window = (
            signal[
                start:end
            ].copy()
        )


        # ----------------------------------------------------
        # Remove mean
        # ----------------------------------------------------

        window -= np.mean(
            window
        )


        # ----------------------------------------------------
        # Hann window
        # ----------------------------------------------------

        hann = np.hanning(
            len(window)
        )

        windowed_signal = (
            window * hann
        )


        # ----------------------------------------------------
        # FFT
        # ----------------------------------------------------

        fft_values = np.fft.rfft(

            windowed_signal,

            n=fft_size
        )


        frequencies = (
            np.fft.rfftfreq(

                fft_size,

                d=1 / fps
            )
        )


        magnitude = np.abs(
            fft_values
        )


        # ----------------------------------------------------
        # Keep heart-rate frequencies only
        # ----------------------------------------------------

        valid = (

            (frequencies >= low_hz)

            &

            (frequencies <= high_hz)

        )


        valid_frequencies = (
            frequencies[valid]
        )

        valid_magnitude = (
            magnitude[valid]
        )


        if len(
            valid_magnitude
        ) == 0:

            continue


        # ----------------------------------------------------
        # Strongest frequency
        # ----------------------------------------------------

        peak_index = np.argmax(

            valid_magnitude
        )


        dominant_frequency = (

            valid_frequencies[
                peak_index
            ]

        )


        peak_magnitude = (

            valid_magnitude[
                peak_index
            ]

        )


        # ----------------------------------------------------
        # Convert frequency to BPM
        # ----------------------------------------------------

        bpm = (
            dominant_frequency
            * 60
        )


        # ----------------------------------------------------
        # Spectral peak strength
        # ----------------------------------------------------

        average_magnitude = (

            np.mean(
                valid_magnitude
            )

        )


        if average_magnitude > 1e-8:

            peak_strength = (

                peak_magnitude
                / average_magnitude

            )

        else:

            peak_strength = 0


        results.append({

            "start_time":
                start / fps,

            "end_time":
                end / fps,

            "bpm":
                bpm,

            "frequency":
                dominant_frequency,

            "peak_strength":
                peak_strength
        })


    return results


print()

print(
    "Analyzing forehead BPM windows..."
)


forehead_bpm_results = (
    analyze_bpm_windows(

        forehead_filtered,

        fps,

        BPM_WINDOW_SECONDS,

        BPM_STEP_SECONDS,

        LOW_HZ,

        HIGH_HZ,

        FFT_SIZE
    )
)


print()
print("=" * 60)
print(
    "              WINDOWED BPM RESULTS"
)
print("=" * 60)


if len(
    forehead_bpm_results
) == 0:

    print(
        "No valid BPM windows."
    )

else:

    for result in forehead_bpm_results:

        print(

            f"{result['start_time']:.1f}s - "
            f"{result['end_time']:.1f}s : "

            f"{result['bpm']:.2f} BPM | "

            f"{result['frequency']:.3f} Hz | "

            f"Peak strength: "

            f"{result['peak_strength']:.2f}"

        )


# ============================================================
# 25. BPM CONSISTENCY
# ============================================================

if len(
    forehead_bpm_results
) > 0:

    bpm_values = np.array([

        result["bpm"]

        for result
        in forehead_bpm_results

    ])


    mean_bpm = np.mean(
        bpm_values
    )

    median_bpm = np.median(
        bpm_values
    )

    bpm_std = np.std(
        bpm_values
    )

    bpm_range = (

        np.max(bpm_values)

        -

        np.min(bpm_values)

    )

else:

    bpm_values = np.array(
        [],
        dtype=float
    )

    mean_bpm = np.nan

    median_bpm = np.nan

    bpm_std = np.nan

    bpm_range = np.nan


print()
print("=" * 60)
print(
    "              BPM CONSISTENCY"
)
print("=" * 60)


if len(bpm_values) > 0:

    print(
        "Mean BPM:",
        round(
            mean_bpm,
            2
        )
    )

    print(
        "Median BPM:",
        round(
            median_bpm,
            2
        )
    )

    print(
        "BPM standard deviation:",
        round(
            bpm_std,
            2
        )
    )

    print(
        "BPM range:",
        round(
            bpm_range,
            2
        )
    )

else:

    print(
        "No BPM estimate available."
    )


# ============================================================
# 26. REGIONAL BPM FEATURES
# ============================================================

def get_region_bpm(
    filtered_signal,
    fps
):
    """
    Calculate the median BPM across the same
    overlapping windows used for the forehead.

    This keeps the regional BPM calculation
    consistent across all facial regions.
    """

    results = analyze_bpm_windows(

        filtered_signal,

        fps,

        BPM_WINDOW_SECONDS,

        BPM_STEP_SECONDS,

        LOW_HZ,

        HIGH_HZ,

        FFT_SIZE
    )


    if len(results) == 0:
        return np.nan


    values = np.array([

        result["bpm"]

        for result in results

    ])


    return float(
        np.median(values)
    )


forehead_region_bpm = (
    get_region_bpm(
        forehead_filtered,
        fps
    )
)

left_region_bpm = (
    get_region_bpm(
        left_cheek_filtered,
        fps
    )
)

right_region_bpm = (
    get_region_bpm(
        right_cheek_filtered,
        fps
    )
)


# ------------------------------------------------------------
# Regional BPM statistics
# ------------------------------------------------------------

regional_bpms = np.array([

    forehead_region_bpm,

    left_region_bpm,

    right_region_bpm

], dtype=float)


valid_regional_bpms = regional_bpms[
    ~np.isnan(regional_bpms)
]


if len(
    valid_regional_bpms
) > 0:

    regional_bpm_std = float(
        np.std(
            valid_regional_bpms
        )
    )

    regional_bpm_range = float(

        np.max(
            valid_regional_bpms
        )

        -

        np.min(
            valid_regional_bpms
        )

    )

else:

    regional_bpm_std = np.nan

    regional_bpm_range = np.nan


# ============================================================
# 27. PEAK STRENGTH FEATURES
# ============================================================

if len(
    forehead_bpm_results
) > 0:

    peak_strength_values = np.array([

        result["peak_strength"]

        for result
        in forehead_bpm_results

    ])


    mean_peak_strength = float(
        np.mean(
            peak_strength_values
        )
    )

    peak_strength_std = float(
        np.std(
            peak_strength_values
        )
    )

else:

    mean_peak_strength = np.nan

    peak_strength_std = np.nan


# ============================================================
# 28. TEMPORAL VARIABILITY
# ============================================================

if len(forehead_filtered) > 1:

    signal_difference = np.diff(
        forehead_filtered
    )

    temporal_variability = float(
        np.std(
            signal_difference
        )
    )

else:

    temporal_variability = np.nan


# ============================================================
# 29. FINAL FEATURE DICTIONARY
# ============================================================

features = {

    "video":
        video_path,

    "duration_seconds":
        float(duration_seconds),

    "fps":
        float(fps),

    "face_detection_rate":
        float(detection_rate),

    "median_bpm":
        (
            float(median_bpm)
            if not np.isnan(median_bpm)
            else np.nan
        ),

    "mean_bpm":
        (
            float(mean_bpm)
            if not np.isnan(mean_bpm)
            else np.nan
        ),

    "bpm_std":
        (
            float(bpm_std)
            if not np.isnan(bpm_std)
            else np.nan
        ),

    "bpm_range":
        (
            float(bpm_range)
            if not np.isnan(bpm_range)
            else np.nan
        ),

    "mean_peak_strength":
        (
            float(mean_peak_strength)
            if not np.isnan(mean_peak_strength)
            else np.nan
        ),

    "peak_strength_std":
        (
            float(peak_strength_std)
            if not np.isnan(peak_strength_std)
            else np.nan
        ),

    "forehead_region_bpm":
        (
            float(forehead_region_bpm)
            if not np.isnan(
                forehead_region_bpm
            )
            else np.nan
        ),

    "left_region_bpm":
        (
            float(left_region_bpm)
            if not np.isnan(
                left_region_bpm
            )
            else np.nan
        ),

    "right_region_bpm":
        (
            float(right_region_bpm)
            if not np.isnan(
                right_region_bpm
            )
            else np.nan
        ),

    "regional_bpm_std":
        (
            float(regional_bpm_std)
            if not np.isnan(
                regional_bpm_std
            )
            else np.nan
        ),

    "regional_bpm_range":
        (
            float(regional_bpm_range)
            if not np.isnan(
                regional_bpm_range
            )
            else np.nan
        ),

    "temporal_variability":
        (
            float(temporal_variability)
            if not np.isnan(
                temporal_variability
            )
            else np.nan
        ),

    # --------------------------------------------------------
    # Cross-region correlation features
    # --------------------------------------------------------

    "forehead_left_corr":
        (
            float(forehead_left_corr)
            if not np.isnan(
                forehead_left_corr
            )
            else np.nan
        ),

    "forehead_right_corr":
        (
            float(forehead_right_corr)
            if not np.isnan(
                forehead_right_corr
            )
            else np.nan
        ),

    "left_right_corr":
        (
            float(left_right_corr)
            if not np.isnan(
                left_right_corr
            )
            else np.nan
        ),

    "regional_coherence":
        (
            float(regional_coherence)
            if not np.isnan(
                regional_coherence
            )
            else np.nan
        )
}


# ============================================================
# 30. PRINT FEATURE VECTOR
# ============================================================

print()
print("=" * 60)
print("              EXTRACTED FEATURES")
print("=" * 60)

for name, value in features.items():

    print(
        f"{name:<30} : {value}"
    )


# ============================================================
# 31. PLOT BPM OVER TIME
# ============================================================

if len(
    forehead_bpm_results
) > 0:

    bpm_times = [

        (
            result["start_time"]
            +
            result["end_time"]
        ) / 2

        for result
        in forehead_bpm_results

    ]


    bpm_values_plot = [

        result["bpm"]

        for result
        in forehead_bpm_results

    ]


    plt.figure(
        figsize=(12, 5)
    )


    plt.plot(

        bpm_times,

        bpm_values_plot,

        marker="o"
    )


    if not np.isnan(median_bpm):

        plt.axhline(

            median_bpm,

            linestyle="--",

            label=(
                f"Median BPM: "
                f"{median_bpm:.1f}"
            )

        )


    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Estimated BPM"
    )

    plt.title(
        "PulseGuard - Windowed BPM Stability"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# ============================================================
# 32. VIDEO QUALITY INFORMATION
# ============================================================

fft_resolution = (
    fps
    / FFT_SIZE
)

bpm_resolution = (
    fft_resolution
    * 60
)


print()
print("=" * 60)
print(
    "                 VIDEO QUALITY"
)
print("=" * 60)

print(
    "Duration:",
    round(
        duration_seconds,
        2
    ),
    "seconds"
)

print(
    "FPS:",
    round(
        fps,
        2
    )
)

print(
    "Face detection:",
    round(
        detection_rate,
        2
    ),
    "%"
)

print(
    "FFT frequency grid:",
    round(
        fft_resolution,
        4
    ),
    "Hz"
)

print(
    "FFT BPM grid:",
    round(
        bpm_resolution,
        2
    ),
    "BPM"
)


# ============================================================
# 33. PIPELINE AUDIT
# ============================================================

print()
print("=" * 60)
print(
    "                 PIPELINE AUDIT"
)
print("=" * 60)


audit = {}


# ------------------------------------------------------------
# Video
# ------------------------------------------------------------

audit["Video opened"] = True


# ------------------------------------------------------------
# MediaPipe
# ------------------------------------------------------------

audit["MediaPipe initialized"] = True


# ------------------------------------------------------------
# Face detection
# ------------------------------------------------------------

audit["Face detection"] = (
    detection_rate >= 80
)


# ------------------------------------------------------------
# ROI extraction
# ------------------------------------------------------------

audit["Forehead ROI extraction"] = (
    len(forehead_rgb) > 0
)

audit["Left cheek ROI extraction"] = (
    len(left_cheek_rgb) > 0
)

audit["Right cheek ROI extraction"] = (
    len(right_cheek_rgb) > 0
)


# ------------------------------------------------------------
# RGB extraction
# ------------------------------------------------------------

audit["RGB extraction"] = (
    forehead_rgb.shape[0] > 0
)


# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------

normalization_means = (
    np.mean(
        forehead_normalized,
        axis=0
    )
)

normalization_stds = (
    np.std(
        forehead_normalized,
        axis=0
    )
)


normalization_pass = (

    np.all(
        np.abs(
            normalization_means
        ) < 0.01
    )

    and

    np.all(
        np.abs(
            normalization_stds - 1
        ) < 0.01
    )

)


audit[
    "RGB normalization"
] = normalization_pass


# ------------------------------------------------------------
# POS
# ------------------------------------------------------------

audit[
    "POS rPPG extraction"
] = (
    len(forehead_pulse) > 0
)


# ------------------------------------------------------------
# Filtering
# ------------------------------------------------------------

audit[
    "Bandpass filtering"
] = (
    len(forehead_filtered) > 0
)


# ------------------------------------------------------------
# BPM
# ------------------------------------------------------------

audit[
    "FFT / BPM estimation"
] = (
    len(
        forehead_bpm_results
    ) > 0
)


# ------------------------------------------------------------
# BPM consistency
# ------------------------------------------------------------

if len(bpm_values) > 0:

    audit[
        "BPM consistency"
    ] = (
        bpm_std < 20
    )

else:

    audit[
        "BPM consistency"
    ] = False


for name, passed in audit.items():

    if passed:

        print(
            "PASS",
            name
        )

    else:

        print(
            "FAIL",
            name
        )


# ============================================================
# 34. SAVE FEATURES
# ============================================================

with open(
    "features.json",
    "w"
) as f:

    json.dump(
        features,
        f,
        indent=4,
        allow_nan=True
    )


print()

print(
    "Features saved to features.json"
)


# ============================================================
# 35. SIGNAL QUALITY INTERPRETATION
# ============================================================

print()
print("=" * 60)
print(
    "              SIGNAL INTERPRETATION"
)
print("=" * 60)


if len(bpm_values) == 0:

    signal_quality = (
        "INCONCLUSIVE"
    )

elif bpm_std < 8:

    signal_quality = (
        "GOOD"
    )

elif bpm_std < 15:

    signal_quality = (
        "MODERATE"
    )

else:

    signal_quality = (
        "POOR"
    )


print(
    "Forehead signal quality:",
    signal_quality
)


if len(bpm_values) > 0:

    print(
        "Representative BPM:",
        round(
            median_bpm,
            2
        )
    )


print()

print(
    "IMPORTANT:"
)

print(
    "These results are physiological-signal"
)

print(
    "features, NOT a final deepfake verdict."
)


# ============================================================
# 36. FRONTEND-STYLE OUTPUT
# ============================================================

if len(bpm_values) > 0:

    frontend_result = {

        "verdict":
            "inconclusive",

        "confidence":
            0.0,

        "bpm":
            round(
                float(median_bpm),
                2
            ),

        "waveform":
            forehead_filtered.tolist(),

        "duration_seconds":
            round(
                duration_seconds,
                2
            ),

        "fps":
            round(
                fps,
                2
            ),

        "message":
            (
                "Physiological signal extracted. "
                "Deepfake classification is not "
                "enabled yet."
            )
    }

else:

    frontend_result = {

        "verdict":
            "inconclusive",

        "confidence":
            0.0,

        "bpm":
            None,

        "waveform":
            [],

        "duration_seconds":
            round(
                duration_seconds,
                2
            ),

        "fps":
            round(
                fps,
                2
            ),

        "message":
            (
                "Could not obtain a reliable "
                "physiological signal."
            )
    }


print()
print("=" * 60)
print(
    "              FRONTEND OUTPUT"
)
print("=" * 60)

print(
    "Verdict:",
    frontend_result["verdict"]
)

print(
    "Confidence:",
    frontend_result["confidence"]
)

print(
    "BPM:",
    frontend_result["bpm"]
)

print(
    "Duration:",
    frontend_result[
        "duration_seconds"
    ]
)

print(
    "FPS:",
    frontend_result["fps"]
)

print(
    "Waveform samples:",
    len(
        frontend_result[
            "waveform"
        ]
    )
)

print(
    "Message:",
    frontend_result["message"]
)


print()
print("=" * 60)
print(
    "             PULSEGUARD COMPLETE"
)
print("=" * 60)
print()