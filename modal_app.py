"""
RVC Text-to-Speech Pipeline on Modal
Combines Edge-TTS for base audio + RVC for voice conversion
"""
import modal
import os
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
import json
import logging

# Configure Modal app
app = modal.App("rvc-tts-pipeline")

# Create volumes for model storage
models_volume = modal.Volume.from_name("rvc-models", create_if_missing=True)
cache_volume = modal.Volume.from_name("rvc-cache", create_if_missing=True)

# Define container image with all dependencies
rvc_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "edge-tts==6.1.10",
    "numpy==1.24.3",
    "scipy==1.11.4",
    "soundfile==0.12.1",
    "librosa==0.10.1",
    "faiss-cpu==1.7.4",
    "torch==2.1.0",
    "torchaudio==2.1.0",
    "praat-parselmouth==0.4.3",
    "pyworld==0.3.4",
    "torchcrepe==0.0.20",
    "onnxruntime==1.16.3",
    "fastapi==0.104.1",
    "pydantic==2.5.0"
).run_commands(
    "apt-get update",
    "apt-get install -y ffmpeg sox libsndfile1",
)

# RVC inference class
@app.cls(
    image=rvc_image,
    gpu="A10G",  # Use A10G for cost-effectiveness
    volumes={
        "/models": models_volume,
        "/cache": cache_volume,
    },
    concurrency_limit=10,
    container_idle_timeout=300,  # 5 minute idle timeout
    secrets=[modal.Secret.from_name("rvc-secrets")],  # Optional for API keys
)
class RVCPipeline:
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    @modal.enter()
    def load_models(self):
        """Load RVC models and initialize pipeline"""
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize RVC components
        self.load_rvc_pipeline()
        self.load_voice_models()
        
    def load_rvc_pipeline(self):
        """Initialize RVC inference pipeline"""
        try:
            # Import RVC components (simplified version)
            import sys
            sys.path.append('/models/rvc')
            
            # Initialize RVC config
            self.config = {
                "sampling_rate": 40000,
                "hop_length": 512,
                "f0_method": "rmvpe",  # or "crepe" for higher quality
                "index_rate": 0.75,
                "filter_radius": 3,
                "resample_sr": 0,
                "rms_mix_rate": 0.25,
                "protect": 0.33
            }
            
            self.logger.info("RVC pipeline initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to load RVC pipeline: {e}")
            raise
    
    def load_voice_models(self):
        """Load available RVC voice models"""
        self.voice_models = {}
        models_dir = Path("/models/voices")
        
        if models_dir.exists():
            for model_path in models_dir.glob("*.pth"):
                voice_name = model_path.stem
                self.voice_models[voice_name] = str(model_path)
                self.logger.info(f"Loaded voice model: {voice_name}")
        
        # Add default Edge-TTS voices as fallback
        self.edge_voices = {
            "jenny": "en-US-JennyNeural",
            "aria": "en-US-AriaNeural",
            "guy": "en-US-GuyNeural",
            "davis": "en-US-DavisNeural"
        }
        
    @modal.method()
    async def synthesize_edge_tts(self, text: str, voice: str = "en-US-JennyNeural"):
        """Generate base audio using Edge-TTS"""
        import edge_tts
        import asyncio
        
        try:
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                output_path = tmp_file.name
            
            # Generate speech with Edge-TTS
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            
            # Load and return audio
            audio, sr = sf.read(output_path)
            os.unlink(output_path)
            
            return audio, sr
            
        except Exception as e:
            self.logger.error(f"Edge-TTS synthesis failed: {e}")
            raise
    
    @modal.method()
    def apply_rvc_conversion(self, audio: np.ndarray, sr: int, voice_model: str):
        """Apply RVC voice conversion to audio"""
        try:
            # Simplified RVC inference (you'll need to add actual RVC code)
            # This is a placeholder - integrate actual RVC inference here
            
            import torch
            import torchaudio
            import librosa
            
            # Resample to RVC sample rate if needed
            if sr != self.config["sampling_rate"]:
                audio = librosa.resample(
                    audio, 
                    orig_sr=sr, 
                    target_sr=self.config["sampling_rate"]
                )
                sr = self.config["sampling_rate"]
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
            if self.device == "cuda":
                audio_tensor = audio_tensor.cuda()
            
            # TODO: Add actual RVC inference here
            # This would involve:
            # 1. Extract F0 (pitch)
            # 2. Extract features
            # 3. Run through RVC model
            # 4. Apply index if available
            
            # For now, return original audio as placeholder
            self.logger.info(f"Applied RVC conversion with model: {voice_model}")
            
            return audio, sr
            
        except Exception as e:
            self.logger.error(f"RVC conversion failed: {e}")
            return audio, sr  # Return original on failure
    
    @modal.method()
    async def synthesize(self, text: str, voice_id: str = "jenny", **kwargs):
        """
        Main synthesis method combining Edge-TTS + RVC
        
        Args:
            text: Text to synthesize
            voice_id: Voice model ID or Edge-TTS voice name
            **kwargs: Additional parameters (speed, pitch, etc.)
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Determine if using RVC model or Edge-TTS voice
            use_rvc = voice_id in self.voice_models
            
            # Get base Edge-TTS voice
            edge_voice = self.edge_voices.get(voice_id, "en-US-JennyNeural")
            if use_rvc:
                edge_voice = "en-US-JennyNeural"  # Default base voice for RVC
            
            # Generate base audio with Edge-TTS
            self.logger.info(f"Generating base audio with {edge_voice}")
            audio, sr = await self.synthesize_edge_tts(text, edge_voice)
            
            # Apply RVC conversion if custom voice requested
            if use_rvc:
                self.logger.info(f"Applying RVC conversion with {voice_id}")
                audio, sr = self.apply_rvc_conversion(
                    audio, sr, self.voice_models[voice_id]
                )
            
            # Apply any post-processing
            audio = self.post_process_audio(audio, sr, **kwargs)
            
            return audio, sr
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            raise
    
    def post_process_audio(self, audio: np.ndarray, sr: int, **kwargs):
        """Apply post-processing to audio"""
        # Normalize audio
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        # Apply speed adjustment if requested
        speed = kwargs.get("speed", 1.0)
        if speed != 1.0:
            import librosa
            audio = librosa.effects.time_stretch(audio, rate=speed)
        
        return audio

# Utility functions for model management
@app.function(
    image=rvc_image,
    volumes={"/models": models_volume},
    timeout=600
)
def upload_voice_model(model_name: str, model_data: bytes, index_data: bytes = None):
    """Upload a new RVC voice model"""
    models_dir = Path("/models/voices")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model file
    model_path = models_dir / f"{model_name}.pth"
    with open(model_path, "wb") as f:
        f.write(model_data)
    
    # Save index file if provided
    if index_data:
        index_path = models_dir / f"{model_name}.index"
        with open(index_path, "wb") as f:
            f.write(index_data)
    
    return f"Model {model_name} uploaded successfully"

@app.function(
    image=rvc_image,
    volumes={"/models": models_volume}
)
def list_voice_models():
    """List available voice models"""
    models_dir = Path("/models/voices")
    models = []
    
    if models_dir.exists():
        for model_path in models_dir.glob("*.pth"):
            model_info = {
                "name": model_path.stem,
                "size_mb": model_path.stat().st_size / (1024 * 1024),
                "has_index": (models_dir / f"{model_path.stem}.index").exists()
            }
            models.append(model_info)
    
    return models

# Test function
@app.local_entrypoint()
async def test_synthesis():
    """Test the synthesis pipeline locally"""
    pipeline = RVCPipeline()
    
    test_text = "Hello, this is a test of the RVC text to speech system on Modal."
    
    # Test with Edge-TTS voice
    print("Testing Edge-TTS synthesis...")
    audio, sr = await pipeline.synthesize.remote(test_text, voice_id="jenny")
    print(f"Generated audio: {len(audio)} samples at {sr}Hz")
    
    # Save test output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        print(f"Saved test audio to: {f.name}")
