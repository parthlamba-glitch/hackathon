import cv2
import os


# ============================================================
#                  REAL VIDEO CLIP EXTRACTOR
# ============================================================


INPUT_FOLDER = "videos/real"

OUTPUT_FOLDER = "videos/real_clips"

CLIP_DURATION = 10


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# GET VIDEOS
# ============================================================

valid_extensions = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)


videos = [

    filename

    for filename
    in os.listdir(INPUT_FOLDER)

    if filename.lower().endswith(
        valid_extensions
    )
]


videos.sort()


print()
print("=" * 60)
print("          PULSEGARD REAL CLIP EXTRACTOR")
print("=" * 60)

print()

print(
    "Input folder:",
    INPUT_FOLDER
)

print(
    "Output folder:",
    OUTPUT_FOLDER
)

print(
    "Clip duration:",
    CLIP_DURATION,
    "seconds"
)

print()

print(
    "Videos found:",
    len(videos)
)

print()


# ============================================================
# PROCESS EACH VIDEO
# ============================================================

total_clips = 0


for filename in videos:

    input_path = os.path.join(
        INPUT_FOLDER,
        filename
    )


    print("-" * 60)

    print(
        "Processing:",
        filename
    )


    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        input_path
    )


    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        continue


    # --------------------------------------------------------
    # Get video information
    # --------------------------------------------------------

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

        print(
            "ERROR: Invalid FPS."
        )

        cap.release()

        continue


    duration = (
        total_frames / fps
    )


    print(
        "FPS:",
        round(
            fps,
            2
        )
    )

    print(
        "Duration:",
        round(
            duration,
            2
        ),
        "seconds"
    )

    print(
        "Resolution:",
        width,
        "x",
        height
    )


    # --------------------------------------------------------
    # Determine number of complete clips
    # --------------------------------------------------------

    number_of_clips = int(
        duration // CLIP_DURATION
    )


    print(
        "Complete 10-second clips:",
        number_of_clips
    )


    if number_of_clips == 0:

        print(
            "Skipping: video shorter than 10 seconds."
        )

        cap.release()

        continue


    # --------------------------------------------------------
    # Create clips
    # --------------------------------------------------------

    base_name = os.path.splitext(
        filename
    )[0]


    for clip_number in range(
        number_of_clips
    ):

        start_frame = int(

            clip_number
            * CLIP_DURATION
            * fps

        )

        frames_to_write = int(

            CLIP_DURATION
            * fps

        )


        # -----------------------------------------------
        # Jump to beginning of clip
        # -----------------------------------------------

        cap.set(

            cv2.CAP_PROP_POS_FRAMES,

            start_frame

        )


        # -----------------------------------------------
        # Output filename
        # -----------------------------------------------

        output_filename = (

            f"{base_name}"
            f"_clip_{clip_number}.mp4"

        )


        output_path = os.path.join(

            OUTPUT_FOLDER,

            output_filename

        )


        # -----------------------------------------------
        # Video writer
        # -----------------------------------------------

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )


        writer = cv2.VideoWriter(

            output_path,

            fourcc,

            fps,

            (
                width,
                height
            )

        )


        if not writer.isOpened():

            print(
                "ERROR: Could not create:",
                output_filename
            )

            continue


        # -----------------------------------------------
        # Write frames
        # -----------------------------------------------

        frames_written = 0


        while (
            frames_written
            < frames_to_write
        ):

            ret, frame = (
                cap.read()
            )


            if not ret:
                break


            writer.write(
                frame
            )


            frames_written += 1


        writer.release()


        # -----------------------------------------------
        # Validate clip
        # -----------------------------------------------

        if (
            frames_written
            == frames_to_write
        ):

            print(
                "Created:",
                output_filename
            )

            total_clips += 1


        else:

            print(
                "WARNING: Incomplete clip:",
                output_filename
            )


            if os.path.exists(
                output_path
            ):

                os.remove(
                    output_path
                )


    cap.release()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("             CLIP EXTRACTION COMPLETE")
print("=" * 60)

print()

print(
    "Total 10-second REAL clips:",
    total_clips
)

print()

print(
    "Saved to:",
    OUTPUT_FOLDER
)

print()