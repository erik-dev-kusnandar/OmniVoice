import os
import torch
import numpy as np
import logging
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import scipy.io.wavfile as wavfile
from typing import Optional, List

from omnivoice import OmniVoice, OmniVoiceGenerationConfig

import asyncio
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model & device config
model = None
# Force to CPU to save VRAM (OmniVoice needs ~10GB on GPU)
device = "cuda:0" 
model_lock = asyncio.Lock()

# Simple in-memory history
generation_history: List[dict] = []

# Output directory for persistent audio files
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    checkpoint = os.getenv("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
    logger.info(f"Loading OmniVoice model from {checkpoint} on {device}...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=False # Disable ASR to save 2-3GB VRAM
    )
    logger.info("OmniVoice Model loaded successfully! ✅")
    yield
    # Cleanup if needed
    logger.info("Shutting down OmniVoice API...")

app = FastAPI(title="OmniVoice REST API", lifespan=lifespan)

# Mount static folder for audio outputs
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    text: str
    language: Optional[str] = "id" # Default to 'id' (Indonesian)
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None
    instruct: Optional[str] = None
    speed: float = 1.0
    num_step: int = 32
    guidance_scale: float = 2.0

# Language Mapping
LANG_MAP = {
    "id": "Indonesian",
    "indonesia": "Indonesian",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OmniVoice API - Speaker Ready</title>
        <style>
            body { 
                margin: 0; padding: 0; 
                background: #0f172a; 
                color: #f8fafc; 
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                display: flex; flex-direction: column; align-items: center; justify-content: center;
                height: 100vh; text-align: center;
            }
            .container {
                background: rgba(30, 41, 59, 0.5);
                padding: 3rem; border-radius: 1.5rem;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
                max-width: 600px;
            }
            h1 { 
                font-size: 3rem; margin-bottom: 1rem;
                background: linear-gradient(to right, #38bdf8, #818cf8);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            }
            p { color: #94a3b8; font-size: 1.2rem; line-height: 1.6; }
            .badge {
                display: inline-block; padding: 0.5rem 1rem;
                background: rgba(34, 197, 94, 0.2);
                color: #4ade80; border-radius: 9999px;
                font-weight: 600; font-size: 0.875rem;
                margin-bottom: 1.5rem;
            }
            .endpoint {
                margin-top: 2rem; padding: 1rem;
                background: #020617; border-radius: 0.75rem;
                font-family: monospace; color: #38bdf8;
                border-left: 4px solid #38bdf8;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="badge">● Server Online</div>
            <h1>OmniVoice API</h1>
            <p>Sistem Voice Cloning & TTS mutakhir siap melayani permintaan narasi kamu.</p>
            <div class="endpoint">POST /generate</div>
            <p style="font-size: 0.9rem; margin-top: 1rem;">Gunakan method <b>POST</b> dengan JSON body untuk mulai.</p>
        </div>
    </body>
    </html>
    """

async def process_generation(
    text: str, 
    language: str = "id", 
    ref_audio_path: str = None, 
    ref_text: str = None, 
    speed: float = 1.0, 
    num_step: int = 32, 
    guidance_scale: float = 2.0,
    instruct: str = None
):
    if not model:
        raise HTTPException(503, "Model not loaded")
    
    try:
        gen_config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=guidance_scale,
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=True
        )

        # Resolve language
        target_lang = language.lower() if language else "indonesian"
        target_lang = LANG_MAP.get(target_lang, target_lang)

        kw = {
            "text": text.strip(),
            "language": target_lang if target_lang != "Auto" else None,
            "generation_config": gen_config,
            "speed": speed
        }

        if ref_audio_path and os.path.exists(ref_audio_path):
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=ref_text
            )
        
        if instruct:
            kw["instruct"] = instruct

        logger.info(f"Generating audio for text: {text[:50]}...")
        async with model_lock:
            audio = await asyncio.to_thread(model.generate, **kw)
        
        # Ensure it's a numpy array on CPU
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        
        if isinstance(audio, (list, tuple)):
            audio = audio[0] # Take first channel or first item
            
        logger.info(f"Audio generated, shape: {getattr(audio, 'shape', 'unknown')}, type: {type(audio)}")
        
        # Convert to 16-bit PCM WAV
        # Force to float32 first for safe normalization
        audio_float = audio.astype(np.float32)
        
        # Normalize to [-1, 1]
        max_val = np.abs(audio_float).max()
        if max_val > 1.0:
            audio_float = audio_float / max_val
        
        # Convert to int16
        waveform = (audio_float * 32767).astype(np.int16)
        
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.wav"
        file_path = OUTPUT_DIR / filename
        
        # Ensure sampling_rate is int
        sr = int(model.sampling_rate)
        wavfile.write(str(file_path), sr, waveform)
        
        with open(file_path, "rb") as f:
            content = f.read()
        
        # We don't remove the file anymore so it can be accessed via URL
        
        return Response(
            content=content, 
            media_type="audio/wav",
            headers={"X-Audio-URL": f"/outputs/{filename}"}
        ), filename

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(500, str(e))

@app.get("/generate")
async def list_history():
    """Tampilkan riwayat generate (metadata saja)"""
    return {
        "total": len(generation_history),
        "history": generation_history[-20:] # Tampilkan 20 terakhir
    }

@app.post("/generate")
async def generate_post(req: GenerateRequest):
    response, filename = await process_generation(
        text=req.text,
        language=req.language,
        ref_audio_path=req.ref_audio_path,
        ref_text=req.ref_text,
        speed=req.speed,
        num_step=req.num_step,
        guidance_scale=req.guidance_scale,
        instruct=req.instruct
    )
    
    # Masih catat ke history (metadata saja)
    generation_history.append({
        "id": filename.replace(".wav", ""),
        "text": req.text,
        "language": req.language,
        "audio_url": f"/outputs/{filename}",
        "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", None, None))
    })
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
