"""
Voice Cloning Agent
Generates ad audio in the podcaster's voice using HuggingFace open TTS models
"""
import os
import torch
import numpy as np
import tempfile
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
from pydub import AudioSegment
import soundfile as sf
import librosa
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

class VoiceCloningAgent:
    def __init__(self):
        """Initialize voice cloning agent with HuggingFace models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Models will be lazy loaded
        self.processor = None
        self.model = None
        self.vocoder = None
        self.speaker_encoder = None
        
    def _load_models(self):
        """Load TTS models"""
        if self.model is None:
            print("Loading SpeechT5 TTS model...")
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                self.vocoder = self.vocoder.to(self.device)
        
        if self.speaker_encoder is None:
            print("Loading Speaker Encoder (SpeechBrain)...")
            # Use copy strategy to avoid symlink permission errors on Windows
            kwargs = {}
            if os.name == 'nt':
                kwargs['local_strategy'] = LocalStrategy.COPY
                
            self.speaker_encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="tmp_model",
                run_opts={"device": self.device},
                **kwargs
            )
    
    def extract_speaker_embedding(self, audio_path: str) -> torch.Tensor:
        """
        Extract speaker embedding from audio sample for voice cloning
        Uses SpeechBrain x-vector model
        """
        print(f"Extracting speaker embedding from: {audio_path}")
        
        # Load audio with soundfile directly to avoid torchaudio backend issues
        signal_np, fs = sf.read(audio_path)
        signal = torch.tensor(signal_np).float()
        
        # Handle channels (soundfile returns (time, channels) or (time,))
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0) # (1, time)
        else:
            signal = signal.transpose(0, 1) # (channels, time)
        
        # Ensure 16k sample rate (required by x-vector model)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
            
        # Ensure mono
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
            
        # Extract embedding
        # SpeechBrain returns (batch, time, channels) -> (1, 1, 512)
        with torch.no_grad():
            embeddings = self.speaker_encoder.encode_batch(signal)
            embeddings = embeddings.squeeze(1)
            
        # Normalize embedding (SpeechT5 expects normalized x-vectors)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        return embeddings
    
    def _create_simple_embedding(self, audio: np.ndarray) -> torch.Tensor:
        """Create a simple speaker embedding from audio features"""
        # Extract basic audio features
        mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=20)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=16000)
        
        # Create a 512-dim embedding (required by SpeechT5)
        embedding = np.zeros(512)
        
        # Fill with MFCC statistics
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        embedding[:20] = mfcc_mean
        embedding[20:40] = mfcc_std
        embedding[40] = np.mean(spectral_centroid)
        embedding[41] = np.std(spectral_centroid)
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    
    def generate_ad_audio(
        self, 
        ad_script: str, 
        speaker_sample_path: str,
        output_path: str = None,
        status_callback = None
    ) -> str:
        """
        Generate ad audio in the speaker's voice
        """
        self._load_models()
        
        if status_callback:
            status_callback(60, "Generating ad audio with voice cloning...")
        
        # Extract speaker embedding
        speaker_embeddings = self.extract_speaker_embedding(speaker_sample_path)
        
        if self.device == "cuda":
            speaker_embeddings = speaker_embeddings.to(self.device)
        
        # Split script into sentences for better synthesis
        sentences = self._split_into_sentences(ad_script)
        
        all_audio = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            print(f"Synthesizing: {sentence[:50]}...")
            
            # Prepare input
            inputs = self.processor(text=sentence, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate speech
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings,
                    vocoder=self.vocoder
                )
            
            audio_np = speech.cpu().numpy()
            all_audio.append(audio_np)
            
            # Add small pause between sentences
            pause = np.zeros(int(16000 * 0.3))  # 0.3 second pause
            all_audio.append(pause)
        
        # Concatenate all audio
        final_audio = np.concatenate(all_audio)
        
        # Save to file
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        sf.write(output_path, final_audio, 16000)
        
        print(f"Generated ad audio saved to: {output_path}")
        return output_path
    
    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences for synthesis"""
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Further split very long sentences
        result = []
        for sentence in sentences:
            if len(sentence) > 200:
                # Split on commas or semicolons
                parts = re.split(r'(?<=[,;])\s+', sentence)
                result.extend(parts)
            else:
                result.append(sentence)
        
        return [s.strip() for s in result if s.strip()]
    
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
