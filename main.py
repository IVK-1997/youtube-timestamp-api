import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.genai import Client
from youtube_transcript_api import YouTubeTranscriptApi

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client(api_key=os.getenv("GEMINI_API_KEY"))


class AskRequest(BaseModel):
    video_url: str
    topic: str


class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


def extract_video_id(url: str) -> str:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1]
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    raise ValueError("Invalid YouTube URL")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        video_id = extract_video_id(request.video_url)

        transcript_data = YouTubeTranscriptApi().fetch(video_id)

        transcript_text = ""
        for entry in transcript_data:
            transcript_text += f"{entry.start}: {entry.text}\n"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
Here is a YouTube transcript with timestamps in seconds.

Find the first timestamp where the topic '{request.topic}' appears.

Return ONLY the timestamp in HH:MM:SS format.

Transcript:
{transcript_text}
"""
        )

        timestamp = response.text.strip()

        return {
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
