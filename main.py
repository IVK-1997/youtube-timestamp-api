import os
import subprocess
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai import Client

app = FastAPI()

client = Client(api_key=os.getenv("GEMINI_API_KEY"))


class AskRequest(BaseModel):
    video_url: str
    topic: str


class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


def download_audio(video_url: str) -> str:
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "-x",
        "--audio-format", "mp3",
        "-o", "audio.%(ext)s",
        video_url,
    ]
    subprocess.run(command, check=True)
    return "audio.mp3"


def upload_and_wait(file_path: str):
    uploaded = client.files.upload(file=file_path)

    while True:
        file_info = client.files.get(name=uploaded.name)

        if file_info.state == "ACTIVE":
            return uploaded

        if file_info.state == "FAILED":
            raise Exception("File processing failed")

        time.sleep(2)


def normalize_timestamp(ts: str) -> str:
    parts = ts.strip().split(":")

    if len(parts) == 3:
        return ts

    if len(parts) == 2:
        return "00:" + ts

    if len(parts) == 1:
        return f"00:00:{parts[0]}"

    raise ValueError("Invalid timestamp format")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    audio_path = None

    try:
        audio_path = download_audio(request.video_url)
        uploaded_file = upload_and_wait(audio_path)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                uploaded_file,
                f"Find the first timestamp where '{request.topic}' is spoken. "
                f"Return ONLY the timestamp in HH:MM:SS format."
            ]
        )

        timestamp = normalize_timestamp(response.text.strip())

        return {
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
