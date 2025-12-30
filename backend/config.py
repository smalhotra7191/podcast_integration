"""
Configuration file for API keys and settings
"""
import os

# ElevenLabs API Configuration
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "sk_41f763477b8923b6d4425abe3b89dba10447bfbc79622c42")

# ElevenLabs Voice Settings
ELEVENLABS_VOICE_SETTINGS = {
    "stability": 0.5,           # 0-1: Lower = more expressive, Higher = more stable
    "similarity_boost": 0.75,   # 0-1: Higher = closer to original voice
    "style": 0.0,               # 0-1: Style exaggeration (only for v2 models)
    "use_speaker_boost": True   # Enhances voice clarity
}

# ElevenLabs Model IDs
ELEVENLABS_MODELS = {
    "multilingual_v2": "eleven_multilingual_v2",  # Best quality, supports 29 languages
    "turbo_v2_5": "eleven_turbo_v2_5",            # Faster, English optimized
    "turbo_v2": "eleven_turbo_v2",                # Fast, good quality
    "monolingual_v1": "eleven_monolingual_v1"     # Legacy English model
}

# Default model for voice cloning
DEFAULT_ELEVENLABS_MODEL = ELEVENLABS_MODELS["multilingual_v2"]
