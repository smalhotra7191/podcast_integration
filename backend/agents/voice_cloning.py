"""
Voice Cloning Agent
Generates ad audio in the podcaster's voice using ElevenLabs API for high-quality
voice cloning with voice similarity scoring using speaker verification
"""
import os
import torch
import numpy as np
import tempfile
import requests
import time
from pydub import AudioSegment
import soundfile as sf
import torchaudio

# Import config for API key
from config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_SETTINGS,
    DEFAULT_ELEVENLABS_MODEL
)

# ElevenLabs API endpoints
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# Lazy import for speechbrain to avoid startup issues
SpeakerRecognition = None
EncoderClassifier = None

def _load_speechbrain_modules():
    """Lazily load speechbrain modules"""
    global SpeakerRecognition, EncoderClassifier
    if SpeakerRecognition is None:
        try:
            from speechbrain.inference.speaker import SpeakerRecognition as SR
            from speechbrain.inference.classifiers import EncoderClassifier as EC
            SpeakerRecognition = SR
            EncoderClassifier = EC
        except ImportError:
            # Fallback for older versions
            from speechbrain.pretrained import EncoderClassifier as EC, SpeakerRecognition as SR
            SpeakerRecognition = SR
            EncoderClassifier = EC


class VoiceSimilarityScorer:
    """
    Computes voice similarity scores between audio files using speaker verification
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.verification_model = None
        
    def _load_model(self):
        """Load speaker verification model"""
        if self.verification_model is None:
            # Load speechbrain modules lazily
            _load_speechbrain_modules()
            
            print("Loading Speaker Verification model...")
            kwargs = {}
            # Import LocalStrategy only when needed
            try:
                from speechbrain.utils.fetching import LocalStrategy
                if os.name == 'nt':
                    kwargs['local_strategy'] = LocalStrategy.COPY
            except ImportError:
                pass
                
            self.verification_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="tmp_model_verification",
                run_opts={"device": self.device},
                **kwargs
            )
    
    def compute_similarity(self, audio_path1: str, audio_path2: str) -> dict:
        """
        Compute similarity score between two audio files
        
        Returns:
            dict with 'score' (0-100), 'is_same_speaker' (bool), 'raw_score' (cosine similarity)
        """
        self._load_model()
        
        try:
            # Load and preprocess audio files
            signal1 = self._load_audio(audio_path1)
            signal2 = self._load_audio(audio_path2)
            
            # Compute similarity using the verification model
            score, prediction = self.verification_model.verify_batch(signal1, signal2)
            
            # Convert score to percentage (0-100)
            raw_score = score.item()
            # The ECAPA-TDNN model outputs cosine similarity scores
            # For zero-shot voice cloning with XTTS-v2:
            # - Raw scores of 0.35-0.50 are typical for good zero-shot cloning
            # - Raw scores of 0.50+ are excellent for zero-shot
            # - Raw scores of 0.65+ require fine-tuning to achieve
            # Realistic normalization for zero-shot TTS:
            if raw_score <= 0.25:
                normalized_score = raw_score * 200  # 0-50%
            elif raw_score <= 0.45:
                # Map 0.25-0.45 to 50-80%
                normalized_score = 50 + (raw_score - 0.25) * 150  # 30% range over 0.20 raw
            elif raw_score <= 0.55:
                # Map 0.45-0.55 to 80-90%
                normalized_score = 80 + (raw_score - 0.45) * 100  # 10% range over 0.10 raw
            else:
                # Map 0.55-1.0 to 90-100%
                normalized_score = 90 + (raw_score - 0.55) * 22.22  # 10% range over 0.45 raw
            normalized_score = max(0, min(100, normalized_score))
            
            return {
                'score': round(normalized_score, 2),
                'is_same_speaker': prediction[0].item(),
                'raw_score': round(raw_score, 4),
                'quality_label': self._get_quality_label(normalized_score)
            }
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return {
                'score': 0,
                'is_same_speaker': False,
                'raw_score': 0,
                'quality_label': 'Error',
                'error': str(e)
            }
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio for speaker verification"""
        signal_np, fs = sf.read(audio_path)
        signal = torch.tensor(signal_np).float()
        
        # Handle channels
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)
        else:
            signal = signal.transpose(0, 1)
        
        # Resample to 16kHz if needed
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
        
        # Ensure mono
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        
        return signal
    
    def _get_quality_label(self, score: float) -> str:
        """Get a human-readable quality label for the similarity score"""
        if score >= 80:
            return "Excellent"
        elif score >= 65:
            return "Good"
        elif score >= 50:
            return "Fair"
        elif score >= 35:
            return "Poor"
        else:
            return "Very Poor"


class VoiceCloningAgent:
    def __init__(self):
        """Initialize voice cloning agent with ElevenLabs API for high-quality voice cloning"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # ElevenLabs API configuration
        self.api_key = ELEVENLABS_API_KEY
        self.base_url = ELEVENLABS_BASE_URL
        self.model_id = DEFAULT_ELEVENLABS_MODEL
        self.voice_settings = ELEVENLABS_VOICE_SETTINGS
        
        # Cloned voice ID (created from speaker sample)
        self.cloned_voice_id = None
        
        self.similarity_scorer = VoiceSimilarityScorer(device=self.device)
        
        # Store last similarity score
        self.last_similarity_score = None
        
    def _get_headers(self):
        """Get API headers for ElevenLabs requests"""
        return {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
    
    def _clone_voice(self, speaker_sample_path: str, voice_name: str = "cloned_voice") -> str:
        """
        Clone a voice using ElevenLabs Instant Voice Cloning API
        
        Args:
            speaker_sample_path: Path to the speaker's audio sample
            voice_name: Name for the cloned voice
            
        Returns:
            voice_id: The ID of the cloned voice
        """
        print(f"Cloning voice from: {speaker_sample_path}")
        
        url = f"{self.base_url}/voices/add"
        
        headers = {
            "Accept": "application/json",
            "xi-api-key": self.api_key
        }
        
        # Prepare the audio file for upload
        with open(speaker_sample_path, 'rb') as audio_file:
            files = [
                ('files', (os.path.basename(speaker_sample_path), audio_file, 'audio/wav'))
            ]
            data = {
                'name': voice_name,
                'description': 'Cloned voice for podcast ad generation'
            }
            
            response = requests.post(url, headers=headers, data=data, files=files)
        
        if response.status_code == 200:
            voice_data = response.json()
            voice_id = voice_data.get('voice_id')
            print(f"Voice cloned successfully! Voice ID: {voice_id}")
            return voice_id
        else:
            raise Exception(f"Failed to clone voice: {response.status_code} - {response.text}")
    
    def _delete_cloned_voice(self, voice_id: str):
        """Delete a cloned voice from ElevenLabs"""
        if not voice_id:
            return
            
        url = f"{self.base_url}/voices/{voice_id}"
        headers = {
            "Accept": "application/json",
            "xi-api-key": self.api_key
        }
        
        try:
            response = requests.delete(url, headers=headers)
            if response.status_code == 200:
                print(f"Deleted cloned voice: {voice_id}")
            else:
                print(f"Warning: Could not delete voice {voice_id}: {response.status_code}")
        except Exception as e:
            print(f"Warning: Error deleting voice: {e}")
    
    def _prepare_speaker_sample(self, audio_path: str) -> str:
        """
        Prepare speaker sample for ElevenLabs voice cloning.
        ElevenLabs works best with clean audio samples.
        
        Args:
            audio_path: Path to the speaker's audio file
            
        Returns:
            Path to the prepared audio file
        """
        print(f"Preparing speaker sample from: {audio_path}")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Resample to 44100 Hz (good quality for ElevenLabs)
        if audio.frame_rate != 44100:
            audio = audio.set_frame_rate(44100)
        
        # Normalize audio
        audio = audio.normalize(headroom=0.1)
        
        # ElevenLabs recommends 1-5 minutes of audio for best results
        # Trim to max 5 minutes if longer
        max_duration_ms = 5 * 60 * 1000  # 5 minutes
        if len(audio) > max_duration_ms:
            audio = audio[:max_duration_ms]
            print(f"Trimmed audio to 5 minutes for optimal voice cloning")
        
        # Export to temp file
        temp_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        audio.export(temp_path, format='wav', parameters=["-acodec", "pcm_s16le"])
        
        print(f"Prepared sample duration: {len(audio)/1000:.1f}s")
        return temp_path
    
    def _generate_speech(self, text: str, voice_id: str) -> bytes:
        """
        Generate speech using ElevenLabs Text-to-Speech API
        
        Args:
            text: Text to synthesize
            voice_id: ElevenLabs voice ID
            
        Returns:
            Audio data as bytes (MP3 format)
        """
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": self.voice_settings
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"TTS failed: {response.status_code} - {response.text}")
    
    def generate_ad_audio(
        self, 
        ad_script: str, 
        speaker_sample_path: str,
        output_path: str = None,
        status_callback = None
    ) -> str:
        """
        Generate ad audio in the speaker's voice using ElevenLabs API
        
        Args:
            ad_script: The ad script text to synthesize
            speaker_sample_path: Path to speaker's audio sample for voice cloning
            output_path: Optional output path for the generated audio
            status_callback: Optional callback for progress updates
            
        Returns:
            Path to the generated audio file
        """
        if status_callback:
            status_callback(60, "Cloning voice with ElevenLabs API...")
        
        # Prepare speaker sample
        prepared_sample = self._prepare_speaker_sample(speaker_sample_path)
        
        # Set output path
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        try:
            # Clone the voice using ElevenLabs Instant Voice Cloning
            voice_name = f"cloned_voice_{int(time.time())}"
            self.cloned_voice_id = self._clone_voice(prepared_sample, voice_name)
            
            if status_callback:
                status_callback(70, "Generating speech with cloned voice...")
            
            print(f"Generating speech for ad script ({len(ad_script)} chars)...")
            
            # Generate speech using the cloned voice
            audio_data = self._generate_speech(ad_script, self.cloned_voice_id)
            
            # Save MP3 to temp file first
            temp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
            with open(temp_mp3, 'wb') as f:
                f.write(audio_data)
            
            # Convert MP3 to WAV for consistency with rest of pipeline
            audio = AudioSegment.from_mp3(temp_mp3)
            audio.export(output_path, format='wav')
            
            # Clean up temp MP3
            try:
                os.unlink(temp_mp3)
            except:
                pass
            
            print(f"Generated audio saved to: {output_path}")
            
        finally:
            # Clean up: delete the cloned voice from ElevenLabs
            if self.cloned_voice_id:
                self._delete_cloned_voice(self.cloned_voice_id)
                self.cloned_voice_id = None
            
            # Clean up prepared sample
            try:
                os.unlink(prepared_sample)
            except:
                pass
        
        # Compute similarity score
        if status_callback:
            status_callback(75, "Computing voice similarity score...")
        
        print("Computing voice similarity score...")
        self.last_similarity_score = self.similarity_scorer.compute_similarity(
            speaker_sample_path, output_path
        )
        print(f"Voice Similarity Score: {self.last_similarity_score['score']}% ({self.last_similarity_score['quality_label']})")
        
        return output_path
    
    def get_similarity_score(self) -> dict:
        """Get the last computed similarity score"""
        return self.last_similarity_score
    
    def create_transition_audio(
        self,
        transition_text: str,
        speaker_sample_path: str,
        output_path: str = None
    ) -> str:
        """
        Create transition audio for smooth ad integration
        """
        return self.generate_ad_audio(
            transition_text,
            speaker_sample_path,
            output_path
        )


class AlternativeVoiceGenerator:
    """
    Alternative voice generation using Bark model (more natural but slower)
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
    
    def _load_models(self):
        """Load Bark model"""
        if self.model is None:
            print("Loading Bark TTS model...")
            from transformers import AutoProcessor, BarkModel
            
            self.processor = AutoProcessor.from_pretrained("suno/bark-small")
            self.model = BarkModel.from_pretrained("suno/bark-small")
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
    
    def generate(self, text: str, voice_preset: str = "v2/en_speaker_6") -> np.ndarray:
        """Generate speech using Bark"""
        self._load_models()
        
        inputs = self.processor(text, voice_preset=voice_preset)
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            audio_array = self.model.generate(**inputs)
        
        return audio_array.cpu().numpy().squeeze()


# Test function
if __name__ == "__main__":
    agent = VoiceCloningAgent()
    print("Voice Cloning Agent initialized successfully!")
    print(f"Using device: {agent.device}")
