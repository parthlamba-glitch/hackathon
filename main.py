import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq


# ============================================================
# PULSEGUARD
# Deepfake Detection using rPPG
#
# CURRENT PIPELINE:
#
# 1. Video loading
# 2. MediaPipe face landmarks
# 3. Forehead + cheek ROIs
# 4. RGB extraction
# 5. RGB normalization
# 6. Windowed POS rPPG extraction
# 7. Bandpass filtering
# 8. FFT
# 9. BPM estimation
# 10. Regional consistency
# 11. Pipeline audit
#
# NOTE:
# This version does NOT make a REAL/DEEPFAKE verdict yet.
# ============================================================


# ============================================================
# 0. CONFIGURATION
# ============================================================

video_path = "test_video.mp4"

# Save a few frames so we can visually inspect our ROIs
sample_frames = [0, 150, 300, 450, 600]

# Heart-rate frequency range
LOW_HZ = 0.7
HIGH_HZ = 4.0

# POS window length in seconds
POS_WINDOW_SECONDS = 1.6


# ============================================================
# PIPELINE STATUS TRACKER
#
# This lets us check at the end whether every major block
# actually ran.
# ============================================================

pipeline_status = {
    "Video opened": False,
    "MediaPipe initialized": False,
    "Face detection": False,
    "Forehead ROI extraction": False,
    "Left cheek ROI extraction": False,
    "Right cheek ROI extraction": False,
    "RGB extraction": False,
    "RGB normalization": False,
    "POS rPPG extraction": False,
    "Bandpass filtering": False,
    "FFT / BPM estimation": False,
    "Regional consistency": False
}


# ============================================================
# 1. VIDEO SETUP
# ============================================================

print()
print("============================================================")
print("                 PULSEGUARD STARTING")
print("============================================================")
print()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    print("ERROR: Could not open video.")
    print("Check that the video exists and the filename is correct.")
    exit()

pipeline_status["Video opened"] = True


fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

duration_seconds = (
    total_frames / fps
    if fps > 0
    else 0
)


print("Video opened successfully")
print("FPS:", fps)
print("Total frames:", total_frames)
print("Resolution:", width, "x", height)
print(
    "Duration:",
    round(duration_seconds, 2),
    "seconds"
)

print()


# ============================================================
# 2. MEDIAPIPE FACE LANDMARKER SETUP
# ============================================================

print("Initializing MediaPipe...")

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


model_path = "face_landmarker.task"


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=model_path
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

pipeline_status["MediaPipe initialized"] = True


# ============================================================
# 3. STORAGE
# ============================================================

# Each element will eventually contain:
#
# [R, G, B]
#
# for one video frame.

forehead_rgb = []

left_cheek_rgb = []

right_cheek_rgb = []


# Number of frames processed
processed_frames = 0

# Number of frames where MediaPipe found a face
face_detected_frames = 0


# ============================================================
# 4. PROCESS VIDEO FRAME-BY-FRAME
# ============================================================

print("Processing video...")
print()


while True:

    # --------------------------------------------------------
    # Read one frame
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        break

    processed_frames += 1


    # --------------------------------------------------------
    # Convert BGR → RGB
    #
    # OpenCV uses BGR.
    # MediaPipe expects RGB.
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Convert NumPy image to MediaPipe image
    # --------------------------------------------------------

    mp_image = mp.Image(

        image_format=(
            mp.ImageFormat.SRGB
        ),

        data=rgb_frame
    )


    # --------------------------------------------------------
    # Timestamp
    #
    # MediaPipe VIDEO mode requires timestamps.
    # --------------------------------------------------------

    timestamp_ms = int(

        (
            (processed_frames - 1)
            / fps
        )
        * 1000

    )


    # --------------------------------------------------------
    # Detect face landmarks
    # --------------------------------------------------------

    result = (
        landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )
    )


    # --------------------------------------------------------
    # 5. CHECK FOR FACE
    # --------------------------------------------------------

    if not result.face_landmarks:

        continue


    face_detected_frames += 1

    pipeline_status["Face detection"] = True


    # We only requested one face.
    # Therefore take the first face.

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


    if forehead_roi.size > 0:

        pipeline_status[
            "Forehead ROI extraction"
        ] = True


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


    if left_cheek_roi.size > 0:

        pipeline_status[
            "Left cheek ROI extraction"
        ] = True


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


    if right_cheek_roi.size > 0:

        pipeline_status[
            "Right cheek ROI extraction"
        ] = True


    # ========================================================
    # 9. CHECK ROIS
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
    # OpenCV gives:
    #
    # [B, G, R]
    #
    # Reverse it:
    #
    # [R, G, B]
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


    pipeline_status["RGB extraction"] = True


    # ========================================================
    # 11. DRAW ROI BOXES
    # ========================================================

    # Green = forehead

    cv2.rectangle(

        frame,

        (fx_min, fy_min),

        (fx_max, fy_max),

        (0, 255, 0),

        2

    )


    # Blue = left cheek

    cv2.rectangle(

        frame,

        (lx_min, ly_min),

        (lx_max, ly_max),

        (255, 0, 0),

        2

    )


    # Red = right cheek

    cv2.rectangle(

        frame,

        (rx_min, ry_min),

        (rx_max, ry_max),

        (0, 0, 255),

        2

    )


    # ========================================================
    # 12. SAVE SAMPLE FRAMES
    # ========================================================

    if (

        processed_frames - 1

        in sample_frames

    ):

        cv2.imwrite(

            f"roi_frame_"
            f"{processed_frames - 1}"
            f".jpg",

            frame

        )


# ============================================================
# 13. CLEAN UP VIDEO + MEDIAPIPE
# ============================================================

cap.release()

landmarker.close()


# ============================================================
# 14. BASIC VIDEO RESULTS
# ============================================================

print()
print("============================================================")
print("                    VIDEO RESULTS")
print("============================================================")

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
    round(detection_rate, 2),
    "%"
)


# ============================================================
# 15. CONVERT SIGNALS TO NUMPY ARRAYS
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


print()

print(
    "Forehead RGB shape:",
    forehead_rgb.shape
)

print(
    "Left cheek RGB shape:",
    left_cheek_rgb.shape
)

print(
    "Right cheek RGB shape:",
    right_cheek_rgb.shape
)


# ============================================================
# 16. CHECK SIGNAL LENGTH
# ============================================================

if len(forehead_rgb) < 10:

    print()
    print(
        "ERROR: Not enough forehead samples."
    )

    print(
        "Cannot continue with rPPG processing."
    )

    exit()


# ============================================================
# 17. RGB NORMALIZATION
# ============================================================

def normalize_rgb(rgb_signal):

    """
    Normalize each RGB channel independently.

    Formula:

        normalized =
            (signal - mean) / standard deviation
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

    std[std == 0] = 1


    normalized = (

        rgb_signal - mean

    ) / std


    return normalized


# Normalize all three regions

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


pipeline_status[
    "RGB normalization"
] = True


# ============================================================
# 18. PLOT NORMALIZED RGB
# ============================================================

time = (

    np.arange(
        len(forehead_normalized)
    )

    / fps

)


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

plt.show()


# ============================================================
# 19. POS rPPG EXTRACTION
# ============================================================

def extract_pos_signal(
    rgb_signal,
    fps,
    window_seconds=1.6
):

    """
    Extract an rPPG signal using a windowed POS-style method.

    Input:

        rgb_signal:
            shape = (number_of_frames, 3)

            columns:
                0 = R
                1 = G
                2 = B

        fps:
            frames per second

    Output:

        pulse_signal:
            one-dimensional rPPG signal
    """


    number_of_frames = (
        rgb_signal.shape[0]
    )


    # --------------------------------------------------------
    # Convert window duration to number of frames
    # --------------------------------------------------------

    window_length = int(

        round(
            window_seconds * fps
        )

    )


    if number_of_frames < window_length:

        raise ValueError(

            "Video is too short "
            "for POS processing."

        )


    # --------------------------------------------------------
    # Final signal
    # --------------------------------------------------------

    pulse_signal = np.zeros(
        number_of_frames
    )


    # --------------------------------------------------------
    # Count how many windows contribute
    # to every frame.
    #
    # This allows us to average overlapping
    # windows instead of simply adding them.
    # --------------------------------------------------------

    contribution_count = np.zeros(
        number_of_frames
    )


    # ========================================================
    # PROCESS TEMPORAL WINDOWS
    # ========================================================

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


        # ----------------------------------------------------
        # Extract current RGB window
        # ----------------------------------------------------

        window = rgb_signal[
            start:end
        ].copy()


        # ----------------------------------------------------
        # Normalize each channel by
        # its temporal mean
        # ----------------------------------------------------

        channel_mean = np.mean(

            window,

            axis=0

        )


        channel_mean[
            channel_mean == 0
        ] = 1


        normalized_window = (

            window
            / channel_mean

        )


        # ----------------------------------------------------
        # Separate RGB
        # ----------------------------------------------------

        R = normalized_window[:, 0]

        G = normalized_window[:, 1]

        B = normalized_window[:, 2]


        # ----------------------------------------------------
        # POS projection
        # ----------------------------------------------------

        X = (

            3 * R
            - 2 * G

        )


        Y = (

            1.5 * R
            + G
            - 1.5 * B

        )


        # ----------------------------------------------------
        # Calculate standard deviations
        # ----------------------------------------------------

        std_x = np.std(X)

        std_y = np.std(Y)


        if std_y < 1e-8:

            continue


        # ----------------------------------------------------
        # Adaptive weighting
        # ----------------------------------------------------

        alpha = (
            std_x
            / std_y
        )


        # ----------------------------------------------------
        # Construct pulse signal
        # ----------------------------------------------------

        h = (
            X
            + alpha * Y
        )


        # ----------------------------------------------------
        # Remove DC component
        # ----------------------------------------------------

        h = (
            h
            - np.mean(h)
        )


        # ----------------------------------------------------
        # Normalize window
        # ----------------------------------------------------

        h_std = np.std(h)


        if h_std < 1e-8:

            continue


        h = (
            h
            / h_std
        )


        # ----------------------------------------------------
        # Add window to final signal
        # ----------------------------------------------------

        pulse_signal[
            start:end
        ] += h


        contribution_count[
            start:end
        ] += 1


    # ========================================================
    # AVERAGE OVERLAPPING WINDOWS
    # ========================================================

    valid = (
        contribution_count > 0
    )


    pulse_signal[valid] /= (
        contribution_count[valid]
    )


    # ========================================================
    # FINAL NORMALIZATION
    # ========================================================

    pulse_signal -= np.mean(
        pulse_signal
    )


    final_std = np.std(
        pulse_signal
    )


    if final_std > 1e-8:

        pulse_signal /= final_std


    return pulse_signal


# ============================================================
# 20. EXTRACT rPPG FROM ALL THREE ROIs
# ============================================================

print()
print("Extracting POS rPPG signals...")


try:

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


    pipeline_status[
        "POS rPPG extraction"
    ] = True


except ValueError as error:

    print()
    print(
        "POS ERROR:",
        error
    )

    exit()


# ============================================================
# 21. PLOT RAW POS rPPG SIGNALS
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

plt.show()


# ============================================================
# 22. BANDPASS FILTER
# ============================================================

def bandpass_filter(

    signal,

    fps,

    low_hz=0.7,

    high_hz=4.0,

    order=3

):

    """
    Keep frequencies between
    low_hz and high_hz.
    """

    nyquist = fps / 2


    # Safety check

    if high_hz >= nyquist:

        raise ValueError(

            "High cutoff frequency "
            "must be below Nyquist frequency."

        )


    low = (
        low_hz
        / nyquist
    )


    high = (
        high_hz
        / nyquist
    )


    b, a = butter(

        order,

        [low, high],

        btype="band"

    )


    filtered = filtfilt(

        b,

        a,

        signal

    )


    return filtered


# ============================================================
# 23. APPLY BANDPASS FILTER
# ============================================================

print()
print("Applying bandpass filter...")


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


pipeline_status[
    "Bandpass filtering"
] = True


# ============================================================
# 24. PLOT FILTERED rPPG
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

plt.show()


# ============================================================
# 25. FFT / BPM ESTIMATION
# ============================================================

def estimate_bpm(

    signal,

    fps,

    low_hz=0.7,

    high_hz=4.0

):

    """
    Find the strongest frequency in the
    plausible heart-rate range.

    Returns:

        bpm
        dominant_frequency
        peak_magnitude
    """


    number_of_samples = (
        len(signal)
    )


    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    spectrum = np.abs(

        rfft(signal)

    )


    # --------------------------------------------------------
    # Frequencies corresponding to FFT bins
    # --------------------------------------------------------

    frequencies = rfftfreq(

        number_of_samples,

        d=1 / fps

    )


    # --------------------------------------------------------
    # Keep only plausible heart-rate frequencies
    # --------------------------------------------------------

    valid = (

        (frequencies >= low_hz)

        &

        (frequencies <= high_hz)

    )


    valid_frequencies = (
        frequencies[valid]
    )


    valid_spectrum = (
        spectrum[valid]
    )


    if len(valid_spectrum) == 0:

        return (
            None,
            None,
            None
        )


    # --------------------------------------------------------
    # Find strongest frequency
    # --------------------------------------------------------

    peak_index = np.argmax(

        valid_spectrum

    )


    dominant_frequency = (

        valid_frequencies[
            peak_index
        ]

    )


    peak_magnitude = (

        valid_spectrum[
            peak_index
        ]

    )


    # --------------------------------------------------------
    # Hz → BPM
    # --------------------------------------------------------

    bpm = (
        dominant_frequency
        * 60
    )


    return (

        bpm,

        dominant_frequency,

        peak_magnitude

    )


# ============================================================
# 26. ESTIMATE BPM FOR ALL THREE ROIs
# ============================================================

forehead_bpm, forehead_frequency, forehead_power = (
    estimate_bpm(

        forehead_filtered,

        fps,

        LOW_HZ,

        HIGH_HZ

    )
)


left_bpm, left_frequency, left_power = (
    estimate_bpm(

        left_cheek_filtered,

        fps,

        LOW_HZ,

        HIGH_HZ

    )
)


right_bpm, right_frequency, right_power = (
    estimate_bpm(

        right_cheek_filtered,

        fps,

        LOW_HZ,

        HIGH_HZ

    )
)


pipeline_status[
    "FFT / BPM estimation"
] = True


# ============================================================
# 27. PRINT BPM RESULTS
# ============================================================

print()
print("============================================================")
print("                       BPM RESULTS")
print("============================================================")


print(

    "Forehead:",

    round(forehead_bpm, 2),

    "BPM",

    "| Frequency:",

    round(forehead_frequency, 3),

    "Hz"

)


print(

    "Left cheek:",

    round(left_bpm, 2),

    "BPM",

    "| Frequency:",

    round(left_frequency, 3),

    "Hz"

)


print(

    "Right cheek:",

    round(right_bpm, 2),

    "BPM",

    "| Frequency:",

    round(right_frequency, 3),

    "Hz"

)


# ============================================================
# 28. PLOT FOREHEAD FREQUENCY SPECTRUM
# ============================================================

frequencies = rfftfreq(

    len(forehead_filtered),

    d=1 / fps

)


spectrum = np.abs(

    rfft(forehead_filtered)

)


plt.figure(
    figsize=(12, 5)
)


plt.plot(

    frequencies,

    spectrum

)


plt.xlim(

    0.5,

    4.2

)


plt.xlabel(
    "Frequency (Hz)"
)


plt.ylabel(
    "Magnitude"
)


plt.title(
    "PulseGuard - Forehead Frequency Spectrum"
)


plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 29. REGIONAL CONSISTENCY
# ============================================================

bpms = np.array([

    forehead_bpm,

    left_bpm,

    right_bpm

])


bpm_mean = np.mean(
    bpms
)


bpm_std = np.std(
    bpms
)


bpm_min = np.min(
    bpms
)


bpm_max = np.max(
    bpms
)


bpm_range = (
    bpm_max
    - bpm_min
)


pipeline_status[
    "Regional consistency"
] = True


# ============================================================
# 30. PRINT REGIONAL CONSISTENCY
# ============================================================

print()
print("============================================================")
print("                 REGIONAL CONSISTENCY")
print("============================================================")


print(

    "Mean BPM:",

    round(bpm_mean, 2)

)


print(

    "BPM standard deviation:",

    round(bpm_std, 2)

)


print(

    "BPM range:",

    round(bpm_range, 2)

)


# ------------------------------------------------------------
# IMPORTANT:
#
# We are NOT calling this REAL or DEEPFAKE yet.
#
# This only tells us whether the three facial regions
# produce similar frequency estimates.
# ------------------------------------------------------------


# ============================================================
# 31. VIDEO QUALITY INFORMATION
# ============================================================

print()
print("============================================================")
print("                    VIDEO QUALITY")
print("============================================================")


print(

    "Duration:",

    round(duration_seconds, 2),

    "seconds"

)


print(

    "FPS:",

    round(fps, 2)

)


print(

    "Face detection:",

    round(detection_rate, 2),

    "%"

)


# ------------------------------------------------------------
# Frequency resolution
#
# Approx:
#
# frequency resolution = 1 / duration
# ------------------------------------------------------------

if duration_seconds > 0:

    frequency_resolution = (

        1
        / duration_seconds

    )

    bpm_resolution = (

        frequency_resolution
        * 60

    )


    print(

        "Approx. FFT frequency resolution:",

        round(
            frequency_resolution,
            3
        ),

        "Hz"

    )


    print(

        "Approx. BPM resolution:",

        round(
            bpm_resolution,
            2
        ),

        "BPM"

    )


# ============================================================
# 32. PIPELINE AUDIT
# ============================================================

print()
print()
print("============================================================")
print("                   PIPELINE AUDIT")
print("============================================================")

print()

all_passed = True


for step, status in pipeline_status.items():

    if status:

        print(
            "PASS  ✓  ",
            step
        )

    else:

        print(
            "FAIL  ✗  ",
            step
        )

        all_passed = False


print()


# ============================================================
# 33. FINAL AUDIT RESULT
# ============================================================

if all_passed:

    print(
        "============================================================"
    )

    print(
        "PIPELINE AUDIT: ALL MAJOR BLOCKS PASSED ✓"
    )

    print(
        "============================================================"
    )

else:

    print(
        "============================================================"
    )

    print(
        "PIPELINE AUDIT: SOME BLOCKS DID NOT PASS ✗"
    )

    print(
        "Check the FAIL entries above."
    )

    print(
        "============================================================"
    )


# ============================================================
# 34. FINAL SUMMARY
# ============================================================

print()
print("============================================================")
print("                    PULSEGUARD SUMMARY")
print("============================================================")

print()

print(
    "Video duration:",
    round(duration_seconds, 2),
    "seconds"
)

print(
    "Frames processed:",
    processed_frames
)

print(
    "Face detection:",
    round(detection_rate, 2),
    "%"
)

print(
    "Forehead BPM:",
    round(forehead_bpm, 2)
)

print(
    "Left cheek BPM:",
    round(left_bpm, 2)
)

print(
    "Right cheek BPM:",
    round(right_bpm, 2)
)

print(
    "Mean BPM:",
    round(bpm_mean, 2)
)

print(
    "Regional BPM std:",
    round(bpm_std, 2)
)

print()

print(
    "IMPORTANT:"
)

print(
    "These results are physiological-signal features,"
)

print(
    "NOT a final deepfake verdict."
)

print(
    "============================================================")