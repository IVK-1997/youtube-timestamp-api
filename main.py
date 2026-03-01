import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.genai import Client

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
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
The following is a public YouTube video:

{request.video_url}

Find the first timestamp where the topic '{request.topic}' is spoken.

Return ONLY the timestamp in HH:MM:SS format.
"""
        )

        timestamp = normalize_timestamp(response.text.strip())

        return {
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
