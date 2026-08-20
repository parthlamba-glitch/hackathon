import os
import json
import sys
import tempfile
import subprocess

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
#                    PULSEGARD API
# ============================================================

app = FastAPI(
    title="PulseGuard API",
    description="API bridge for PulseGuard physiological analysis",
    version="1.0"
)


# ============================================================
#                       CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
#                       HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "PulseGuard API"
    }


# ============================================================
#                     ANALYZE VIDEO
# ============================================================

@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:

        return {
            "success": False,
            "error": "No file provided."
        }


    # --------------------------------------------------------
    # Create temporary video file
    # --------------------------------------------------------

    extension = os.path.splitext(
        file.filename
    )[1]

    if not extension:
        extension = ".webm"


    temp_path = None


    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_path = temp_file.name

            contents = await file.read()

            temp_file.write(
                contents
            )


        # ----------------------------------------------------
        # Run main.py
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("             PULSEGARD API REQUEST")
        print("=" * 60)

        print()

        print(
            "Received:",
            file.filename
        )

        print(
            "Temporary file:",
            temp_path
        )

        print()

        print(
            "Running PulseGuard pipeline..."
        )


        process = subprocess.run(

            [
                sys.executable,
                "main.py",
                temp_path
            ],

            capture_output=True,

            text=True
        )


        # ----------------------------------------------------
        # Check main.py
        # ----------------------------------------------------

        if process.returncode != 0:

            print(
                process.stderr
            )

            return {

                "success": False,

                "error":
                    "PulseGuard analysis failed.",

                "details":
                    process.stderr
            }


        # ----------------------------------------------------
        # Check features.json
        # ----------------------------------------------------

        if not os.path.exists(
            "features.json"
        ):

            return {

                "success": False,

                "error":
                    "features.json was not generated."
            }


        # ----------------------------------------------------
        # Run prediction
        # ----------------------------------------------------

        print(
            "Running ML prediction..."
        )


        prediction_process = subprocess.run(

            [
                sys.executable,
                "predict.py"
            ],

            capture_output=True,

            text=True
        )


        # ----------------------------------------------------
        # Check prediction
        # ----------------------------------------------------

        if prediction_process.returncode != 0:

            print(
                prediction_process.stderr
            )

            return {

                "success": False,

                "error":
                    "ML prediction failed.",

                "details":
                    prediction_process.stderr
            }


        # ----------------------------------------------------
        # Parse prediction JSON
        # --------------------------------------------------------

        prediction_output = (
            prediction_process.stdout.strip()
        )


        # predict.py prints other information before JSON,
        # so find the JSON object in its output.

        json_start = prediction_output.find("{")

        json_end = prediction_output.rfind("}")


        if (
            json_start == -1
            or
            json_end == -1
        ):

            return {

                "success": False,

                "error":
                    "Could not parse prediction output.",

                "raw_output":
                    prediction_output
            }


        json_text = prediction_output[
            json_start:json_end + 1
        ]


        prediction = json.loads(
            json_text
        )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        print()

        print(
            "Analysis complete."
        )

        print(
            "Verdict:",
            prediction.get(
                "verdict"
            )
        )

        print()

        return {

            "success": True,

            "result": prediction

        }


    except Exception as e:

        return {

            "success": False,

            "error": str(e)

        }


    finally:

        # ----------------------------------------------------
        # Delete temporary video
        # ----------------------------------------------------

        if (
            temp_path
            and
            os.path.exists(temp_path)
        ):

            os.remove(
                temp_path
            )