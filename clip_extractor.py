import cv2
import os


# ============================================================
#              PULSEGARD CLIP EXTRACTOR
# ============================================================

REAL_INPUT = "videos/real"
FAKE_INPUT = "videos/fake"

REAL_OUTPUT = "videos/real_clips"
FAKE_OUTPUT = "videos/fake_clips"

CLIP_DURATION = 10


# ============================================================
# VALID VIDEO EXTENSIONS
# ============================================================

VALID_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)


# ============================================================
# EXTRACT CLIPS FROM ONE FOLDER
# ============================================================

def extract_clips(
    input_folder,
    output_folder,
    label_name
):

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    os.makedirs(
        output_folder,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Find videos
    # --------------------------------------------------------

    videos = [

        filename

        for filename
        in os.listdir(input_folder)

        if filename.lower().endswith(
            VALID_EXTENSIONS
        )
    ]


    videos.sort()


    print()
    print("=" * 60)

    print(
        label_name,
        "VIDEO CLIP EXTRACTION"
    )

    print("=" * 60)

    print()

    print(
        "Input:",
        input_folder
    )

    print(
        "Output:",
        output_folder
    )

    print(
        "Videos found:",
        len(videos)
    )

    print()


    total_clips = 0


    # ========================================================
    # PROCESS EACH VIDEO
    # ========================================================

    for filename in videos:

        input_path = os.path.join(
            input_folder,
            filename
        )


        print("-" * 60)

        print(
            "Processing:",
            filename
        )


        # ----------------------------------------------------
        # Open video
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            input_path
        )


        if not cap.isOpened():

            print(
                "ERROR: Could not open video."
            )

            continue


        # ----------------------------------------------------
        # Video information
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Number of complete 10-second clips
        # ----------------------------------------------------

        number_of_clips = int(
            duration // CLIP_DURATION
        )


        print(
            "Complete 10-second clips:",
            number_of_clips
        )


        if number_of_clips == 0:

            print(
                "SKIPPED: Video shorter than 10 seconds."
            )

            cap.release()

            continue


        # ----------------------------------------------------
        # Base filename
        # ----------------------------------------------------

        base_name = os.path.splitext(
            filename
        )[0]


        # ====================================================
        # EXTRACT CLIPS
        # ====================================================

        for clip_number in range(
            number_of_clips
        ):

            # ------------------------------------------------
            # Starting frame
            # ------------------------------------------------

            start_frame = int(

                clip_number
                * CLIP_DURATION
                * fps

            )


            # ------------------------------------------------
            # Number of frames
            # ------------------------------------------------

            frames_to_write = int(

                CLIP_DURATION
                * fps

            )


            # ------------------------------------------------
            # Move video position
            # ------------------------------------------------

            cap.set(

                cv2.CAP_PROP_POS_FRAMES,

                start_frame

            )


            # ------------------------------------------------
            # Output filename
            # ------------------------------------------------

            output_filename = (

                f"{base_name}"
                f"_clip_{clip_number}.mp4"

            )


            output_path = os.path.join(

                output_folder,

                output_filename

            )


            # ------------------------------------------------
            # Video writer
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Write frames
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Validate clip
            # ------------------------------------------------

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


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 60)

    print(
        label_name,
        "CLIP EXTRACTION COMPLETE"
    )

    print("=" * 60)

    print()

    print(
        "Total 10-second clips:",
        total_clips
    )

    print(
        "Saved to:",
        output_folder
    )

    print()

    return total_clips


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 60)
print(
    "          PULSEGARD CLIP EXTRACTOR"
)
print("=" * 60)


# ------------------------------------------------------------
# REAL VIDEOS
# ------------------------------------------------------------

real_count = extract_clips(

    REAL_INPUT,

    REAL_OUTPUT,

    "REAL"
)


# ------------------------------------------------------------
# FAKE VIDEOS
# ------------------------------------------------------------

fake_count = extract_clips(

    FAKE_INPUT,

    FAKE_OUTPUT,

    "FAKE"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print(
    "             EXTRACTION SUMMARY"
)
print("=" * 60)

print()

print(
    "REAL clips:",
    real_count
)

print(
    "FAKE clips:",
    fake_count
)

print()

print(
    "Total clips:",
    real_count + fake_count
)

print()