from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import tempfile
import shutil
import os
from main import process_video

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    return {"status": "healthy"}

@app.post("/analyze-tennis/")
async def analyze_tennis_video(video: UploadFile = File(...)):
    """Process video and return results immediately"""
    
    # Validate file type
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(400, detail="Only MP4/AVI/MOV files allowed")

    
    with tempfile.TemporaryDirectory() as temp_dir:
            # Create required directories
        os.makedirs(f"{temp_dir}/input_videos", exist_ok=True)
            
            # Save uploaded video
        video_path = f"{temp_dir}/input_videos/{video.filename}"
        with open(video_path, "wb") as buffer:
            contents = await video.read()  # Read all content first
            buffer.write(contents)
            
            # Process synchronously (blocks until complete)
        result = process_video(video_path)
        safe_result = jsonable_encoder(result)
            
        return JSONResponse(safe_result)

