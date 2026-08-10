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
# 3. STORAGE FOR OUR SIGNAL
# ============================================================

green_values = []
red_values = []
blue_values = []

face_detected_frames = 0
processed_frames = 0
# ============================================================
# 4. PROCESS VIDEO FRAME-BY-FRAME
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    processed_frames += 1

    # OpenCV gives us BGR.
    # MediaPipe expects RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert NumPy image into MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp required for VIDEO mode
    timestamp_ms = int((processed_frames / fps) * 1000)

    # Detect face landmarks
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # --------------------------------------------------------
    # 5. CHECK WHETHER A FACE WAS FOUND
    # --------------------------------------------------------

    if not result.face_landmarks:
        continue

    face_detected_frames += 1

    # Get the first detected face
    landmarks = result.face_landmarks[0]


    # --------------------------------------------------------
    # 6. TEMPORARY FOREHEAD ROI
    # --------------------------------------------------------
    #
    # We will improve this after confirming that
    # face landmarks are working.
    #
    # These are normalized coordinates selected
    # relative to the detected face.
    # --------------------------------------------------------

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

    x_min = max(min(x_coords), 0)
    x_max = min(max(x_coords), width)

    y_min = max(min(y_coords), 0)
    y_max = min(max(y_coords), height)


    # --------------------------------------------------------
    # 7. EXTRACT FOREHEAD ROI
    # --------------------------------------------------------

    roi = frame[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        continue


    # --------------------------------------------------------
    # 8. AVERAGE PIXEL COLOR
    # --------------------------------------------------------

    mean_color = roi.mean(axis=(0, 1))

    # OpenCV uses BGR
    blue = mean_color[0]
    green = mean_color[1]
    red = mean_color[2]

    blue_values.append(blue)
    green_values.append(green)
    red_values.append(red)

# ============================================================
# 9. CLEAN UP
# ============================================================
cap.release()
landmarker.close()
# ============================================================
# 10. PRINT RESULTS
# ============================================================

print()
print("========== RESULTS ==========")

print("Processed frames:", processed_frames)
print("Face detected frames:", face_detected_frames)

if processed_frames > 0:

    detection_rate = (
        face_detected_frames / processed_frames
    ) * 100

    print(
        "Face detection rate:",
        round(detection_rate, 2),
        "%"
    )

print("Signal samples:", len(green_values))


# ============================================================
# 11. PLOT RAW GREEN SIGNAL
# ============================================================

if len(green_values) > 0:

    time = np.arange(len(green_values)) / fps

    plt.figure(figsize=(12, 5))

    plt.plot(
        time,
        green_values
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Average Green Intensity")

    plt.title(
        "PulseGuard - Raw Forehead Green Channel"
    )

    plt.grid(True)

    plt.show()

else:

    print("No signal was extracted.")