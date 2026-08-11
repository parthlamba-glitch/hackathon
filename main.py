import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. VIDEO SETUP
# ============================================================

video_path = "test_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video opened successfully")
print("FPS:", fps)
print("Total frames:", total_frames)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Resolution:", width, "x", height)


# ============================================================
# 2. MEDIAPIPE FACE LANDMARKER SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

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

landmarker = FaceLandmarker.create_from_options(options)


# ============================================================
# 3. STORAGE FOR RGB SIGNALS
# ============================================================

forehead_rgb = []
left_cheek_rgb = []
right_cheek_rgb = []

face_detected_frames = 0
processed_frames = 0

# Frames that we want to save for checking the ROIs
sample_frames = [0, 50, 100, 150]


# ============================================================
# 4. PROCESS VIDEO FRAME-BY-FRAME
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    processed_frames += 1


    # ========================================================
    # 4A. CONVERT BGR → RGB
    # ========================================================

    # OpenCV reads images as BGR.
    # MediaPipe expects RGB.

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # 4B. CREATE MEDIAPIPE IMAGE
    # ========================================================

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ========================================================
    # 4C. CREATE VIDEO TIMESTAMP
    # ========================================================

    timestamp_ms = int(
        ((processed_frames - 1) / fps) * 1000
    )


    # ========================================================
    # 4D. DETECT FACE LANDMARKS
    # ========================================================

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # 5. CHECK WHETHER A FACE WAS FOUND
    # ========================================================

    if not result.face_landmarks:
        continue

    face_detected_frames += 1

    # We only asked MediaPipe to detect one face
    landmarks = result.face_landmarks[0]


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

    fx_min = max(min(x_coords), 0)
    fx_max = min(max(x_coords), width)

    fy_min = max(min(y_coords), 0)
    fy_max = min(max(y_coords), height)

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

    lx_min = max(min(x_coords), 0)
    lx_max = min(max(x_coords), width)

    ly_min = max(min(y_coords), 0)
    ly_max = min(max(y_coords), height)

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

    rx_min = max(min(x_coords), 0)
    rx_max = min(max(x_coords), width)

    ry_min = max(min(y_coords), 0)
    ry_max = min(max(y_coords), height)

    right_cheek_roi = frame[
        ry_min:ry_max,
        rx_min:rx_max
    ]


    # ========================================================
    # 9. CHECK THAT ALL ROIs ARE VALID
    # ========================================================

    if (
        forehead_roi.size == 0
        or left_cheek_roi.size == 0
        or right_cheek_roi.size == 0
    ):
        continue


    # ========================================================
    # 10. CALCULATE AVERAGE RGB VALUES
    # ========================================================

    # OpenCV ROI is BGR.
    # mean() gives:
    #
    # [Blue, Green, Red]
    #
    # We reverse it to:
    #
    # [Red, Green, Blue]

    forehead_mean = forehead_roi.mean(axis=(0, 1))
    left_cheek_mean = left_cheek_roi.mean(axis=(0, 1))
    right_cheek_mean = right_cheek_roi.mean(axis=(0, 1))

    forehead_rgb.append(
        forehead_mean[::-1]
    )

    left_cheek_rgb.append(
        left_cheek_mean[::-1]
    )

    right_cheek_rgb.append(
        right_cheek_mean[::-1]
    )


    # ========================================================
    # 11. DRAW ROI BOXES
    # ========================================================

    # Green box = forehead
    cv2.rectangle(
        frame,
        (fx_min, fy_min),
        (fx_max, fy_max),
        (0, 255, 0),
        2
    )

    # Blue box = left cheek
    cv2.rectangle(
        frame,
        (lx_min, ly_min),
        (lx_max, ly_max),
        (255, 0, 0),
        2
    )

    # Red box = right cheek
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

    if processed_frames - 1 in sample_frames:

        cv2.imwrite(
            f"roi_frame_{processed_frames - 1}.jpg",
            frame
        )


# ============================================================
# 13. CLEAN UP
# ============================================================

cap.release()
landmarker.close()


# ============================================================
# 14. CONVERT SIGNALS TO NUMPY ARRAYS
# ============================================================

forehead_rgb = np.array(
    forehead_rgb
)

left_cheek_rgb = np.array(
    left_cheek_rgb
)

right_cheek_rgb = np.array(
    right_cheek_rgb
)


# ============================================================
# 15. PRINT RESULTS
# ============================================================

print()
print("========== RESULTS ==========")

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
        face_detected_frames /
        processed_frames
    ) * 100

    print(
        "Face detection rate:",
        round(detection_rate, 2),
        "%"
    )

print(
    "Forehead signal shape:",
    forehead_rgb.shape
)

print(
    "Left cheek signal shape:",
    left_cheek_rgb.shape
)

print(
    "Right cheek signal shape:",
    right_cheek_rgb.shape
)


# ============================================================
# 16. NORMALIZE RGB SIGNAL
# ============================================================

def normalize_rgb(rgb_signal):

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


forehead_normalized = normalize_rgb(
    forehead_rgb
)

left_cheek_normalized = normalize_rgb(
    left_cheek_rgb
)

right_cheek_normalized = normalize_rgb(
    right_cheek_rgb
)

# ============================================================
# POS rPPG EXTRACTION
# ============================================================

def extract_pos_signal(rgb_signal, fps):
    """
    Extract an rPPG pulse signal using a windowed POS method.

    Input:
        rgb_signal -> NumPy array of shape (frames, 3)
                      columns = R, G, B

        fps        -> video frame rate

    Output:
        pulse_signal -> 1D NumPy array
    """

    number_of_frames = rgb_signal.shape[0]

    # POS commonly uses a window around 1.6 seconds.
    window_length = int(round(1.6 * fps))

    if number_of_frames < window_length:
        raise ValueError(
            "Video is too short for POS processing."
        )

    pulse_signal = np.zeros(number_of_frames)

    for start in range(
        0,
        number_of_frames - window_length + 1
    ):

        end = start + window_length

        # ----------------------------------------------------
        # Get current RGB window
        # ----------------------------------------------------

        window = rgb_signal[start:end].copy()

        # ----------------------------------------------------
        # Normalize each RGB channel by its mean
        # ----------------------------------------------------

        channel_mean = np.mean(
            window,
            axis=0
        )

        channel_mean[channel_mean == 0] = 1

        normalized_window = (
            window / channel_mean
        )

        # ----------------------------------------------------
        # POS projection
        # ----------------------------------------------------

        R = normalized_window[:, 0]
        G = normalized_window[:, 1]
        B = normalized_window[:, 2]

        X = 3 * R - 2 * G

        Y = 1.5 * R + G - 1.5 * B

        # ----------------------------------------------------
        # Adaptive weighting
        # ----------------------------------------------------

        std_x = np.std(X)
        std_y = np.std(Y)

        if std_y < 1e-8:
            continue

        alpha = std_x / std_y

        h = X + alpha * Y

        # ----------------------------------------------------
        # Normalize this window's pulse signal
        # ----------------------------------------------------

        h = h - np.mean(h)

        h_std = np.std(h)

        if h_std < 1e-8:
            continue

        h = h / h_std

        # ----------------------------------------------------
        # Add this window into the final signal
        # ----------------------------------------------------

        pulse_signal[start:end] += h

    # --------------------------------------------------------
    # Final normalization
    # --------------------------------------------------------

    pulse_signal -= np.mean(pulse_signal)

    std = np.std(pulse_signal)

    if std > 1e-8:
        pulse_signal /= std

    return pulse_signal

# ============================================================
# EXTRACT rPPG FROM ALL THREE ROIs
# ============================================================

forehead_pulse = extract_pos_signal(
    forehead_rgb,
    fps
)

left_cheek_pulse = extract_pos_signal(
    left_cheek_rgb,
    fps
)

right_cheek_pulse = extract_pos_signal(
    right_cheek_rgb,
    fps
)

# ============================================================
# 17. PLOT NORMALIZED FOREHEAD RGB
# ============================================================

if len(forehead_normalized) > 0:

    time = np.arange(
        len(forehead_normalized)
    ) / fps

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

    plt.show()

else:

    print(
        "No forehead signal was extracted."
    )