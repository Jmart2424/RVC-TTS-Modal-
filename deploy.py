"""
FastAPI deployment for n8n integration
Provides HTTP endpoints for TTS synthesis
"""
import modal
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import io
import soundfile as sf
import numpy as np
import logging
import asyncio

# Import the RVC pipeline app
from modal_app import app as rvc_app, RVCPipeline

# Create FastAPI app
web_app = FastAPI(
    title="RVC Text-to-Speech API",
    description="Fast TTS with custom voice cloning via RVC",
    version="1.0.0"
)

# Request/Response models
class SynthesisRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice_id: str = Field(default="jenny", description="Voice model ID")
    speed: float = Field(default=1.0, description="Speech speed (0.5-2.0)")
    output_format: str = Field(default="wav", description="Output format (wav/mp3)")
    stream: bool = Field(default=False, description="Stream audio response")

class VoiceInfo(BaseModel):
    id: str
    name: str
    type: str
    description: Optional[str] = None

# Initialize pipeline reference
pipeline = None

@web_app.on_event("startup")
async def startup_event():
    """Initialize the RVC pipeline on startup"""
    global pipeline
    logging.info("Initializing RVC pipeline...")
    pipeline = RVCPipeline()
    logging.info("Pipeline ready")

@web_app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "RVC-TTS-Modal",
        "version": "1.0.0",
        "endpoints": [
            "/synthesize",
            "/voices",
            "/health"
        ]
    }

@web_app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "gpu_available": True,
        "models_loaded": pipeline is not None
    }

@web_app.post("/synthesize")
async def synthesize(request: SynthesisRequest):
    """
    Main synthesis endpoint for n8n
    Returns audio file directly
    """
    try:
        # Validate input
        if not request.text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 chars)")
        
        # Call the synthesis method
        logging.info(f"Synthesizing text with voice: {request.voice_id}")
        audio, sample_rate = await pipeline.synthesize.remote(
            text=request.text,
            voice_id=request.voice_id,
            speed=request.speed
        )
        
        # Convert numpy array to bytes
        audio_buffer = io.BytesIO()
        
        if request.output_format == "wav":
            sf.write(audio_buffer, audio, sample_rate, format="WAV")
            media_type = "audio/wav"
        else:
            # For MP3, we'd need additional conversion
            sf.write(audio_buffer, audio, sample_rate, format="WAV")
            media_type = "audio/wav"
        
        audio_buffer.seek(0)
        
        # Return audio file directly (not base64)
        if request.stream:
            return StreamingResponse(
                audio_buffer,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"inline; filename=speech.{request.output_format}"
                }
            )
        else:
            return Response(
                content=audio_buffer.read(),
                media_type=media_type,
                headers={
                    "Content-Disposition": f"inline; filename=speech.{request.output_format}"
                }
            )
        
    except Exception as e:
        logging.error(f"Synthesis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/voices")
async def list_voices():
    """
    List available voices
    Includes both Edge-TTS and custom RVC models
    """
    voices = []
    
    # Add Edge-TTS voices
    edge_voices = [
        VoiceInfo(id="jenny", name="Jenny", type="edge-tts", description="US Female"),
        VoiceInfo(id="aria", name="Aria", type="edge-tts", description="US Female"),
        VoiceInfo(id="guy", name="Guy", type="edge-tts", description="US Male"),
        VoiceInfo(id="davis", name="Davis", type="edge-tts", description="US Male"),
    ]
    voices.extend([v.dict() for v in edge_voices])
    
    # Add custom RVC models
    from modal_app import list_voice_models
    custom_models = list_voice_models.remote()
    
    for model in custom_models:
        voices.append({
            "id": model["name"],
            "name": model["name"].replace("_", " ").title(),
            "type": "rvc",
            "description": f"Custom RVC model ({model['size_mb']:.1f}MB)"
        })
    
    return {"voices": voices, "total": len(voices)}

@web_app.post("/upload-voice")
async def upload_voice(name: str, model_file: bytes, index_file: Optional[bytes] = None):
    """
    Upload a new RVC voice model
    """
    try:
        from modal_app import upload_voice_model
        result = upload_voice_model.remote(name, model_file, index_file)
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Deploy the FastAPI app on Modal
@rvc_app.function(
    image=modal.Image.debian_slim(python_version="3.10").pip_install(
        "fastapi==0.104.1",
        "soundfile==0.12.1",
        "numpy==1.24.3"
    ),
    gpu="A10G",
    container_idle_timeout=300,
    concurrency_limit=10
)
@modal.web_endpoint()
def fastapi_app():
    """Deploy FastAPI app on Modal"""
    return web_app

# Alternative ASGI deployment
@rvc_app.function(
    image=modal.Image.debian_slim(python_version="3.10").pip_install(
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "soundfile==0.12.1",
        "numpy==1.24.3"
    ),
    gpu="A10G",
    container_idle_timeout=300,
    allow_concurrent_inputs=10
)
@modal.asgi_app()
def serve():
    """Serve the FastAPI app via ASGI"""
    return web_app
