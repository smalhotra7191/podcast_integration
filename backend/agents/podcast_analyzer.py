"""
Podcast Analyzer Agent
Transcribes and analyzes podcast audio to find optimal ad insertion points
Uses HuggingFace open models for speech-to-text and semantic analysis
"""
import os
import torch
import numpy as np
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from sentence_transformers import SentenceTransformer
from pydub import AudioSegment
import tempfile
import json

class PodcastAnalyzer:
    def __init__(self):
        """Initialize the podcast analyzer with HuggingFace models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Models will be lazy loaded
        self.transcriber = None
        self.sentence_model = None
        
    def _load_transcriber(self):
        """Load Whisper model for transcription"""
        if self.transcriber is None:
            print("Loading Whisper transcription model...")
            # Using OpenAI Whisper through HuggingFace - open source
            self.transcriber = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",  # Smaller model for faster processing
                torch_dtype=self.torch_dtype,
                device=self.device,
                chunk_length_s=30,
                return_timestamps=True
            )
    
    def _load_sentence_model(self):
        """Load sentence transformer for semantic similarity"""
        if self.sentence_model is None:
            print("Loading sentence transformer model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            if self.device == "cuda":
                self.sentence_model = self.sentence_model.to(self.device)
    
    def extract_audio(self, media_path: str) -> str:
        """Extract audio from video file if needed"""
        ext = os.path.splitext(media_path)[1].lower()
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if ext in video_extensions:
            print(f"Extracting audio from video: {media_path}")
            # Use pydub to extract audio
            video = AudioSegment.from_file(media_path)
            
            # Create temp audio file
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            video.export(temp_audio.name, format='wav')
            return temp_audio.name
        
        return media_path
    
    def transcribe(self, audio_path: str, status_callback=None) -> dict:
        """
        Transcribe audio and return timestamped segments
        """
        self._load_transcriber()
        
        if status_callback:
            status_callback(20, "Transcribing podcast audio...")
        
        print(f"Transcribing: {audio_path}")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        duration_seconds = len(audio) / 1000
        
        # Export to wav for transcription
        temp_wav = None
        if not audio_path.endswith('.wav'):
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio.export(temp_wav.name, format='wav', parameters=["-ac", "1", "-ar", "16000"])
            audio_path = temp_wav.name
        
        # Transcribe
        result = self.transcriber(audio_path)
        
        # Clean up temp file
        if temp_wav:
            os.unlink(temp_wav.name)
        
        # Process results
        transcription = {
            'full_text': result['text'],
            'chunks': [],
            'duration': duration_seconds
        }
        
        if 'chunks' in result:
            for chunk in result['chunks']:
                transcription['chunks'].append({
                    'text': chunk['text'],
                    'start': chunk['timestamp'][0] if chunk['timestamp'][0] is not None else 0,
                    'end': chunk['timestamp'][1] if chunk['timestamp'][1] is not None else duration_seconds
                })
        else:
            # Create single chunk if no timestamps
            transcription['chunks'] = [{
                'text': result['text'],
                'start': 0,
                'end': duration_seconds
            }]
        
        return transcription
    
    def analyze_content(self, transcription: dict, ad_analysis: dict, status_callback=None) -> dict:
        """
        Analyze podcast content to find sections related to ad topic
        """
        self._load_sentence_model()
        
        if status_callback:
            status_callback(40, "Analyzing podcast content for ad placement...")
        
        # Create a topic description from ad analysis
        ad_topic = f"{ad_analysis.get('problem_solved', '')} {ad_analysis.get('product_description', '')}"
        keywords = ad_analysis.get('keywords', [])
        
        # Embed the ad topic
        ad_embedding = self.sentence_model.encode(ad_topic, convert_to_tensor=True)
        
        # Create segments (group chunks into ~30 second segments)
        segments = self._create_segments(transcription['chunks'], target_duration=30)
        
        # Score each segment for relevance
        scored_segments = []
        for segment in segments:
            segment_text = segment['text']
            
            # Semantic similarity
            segment_embedding = self.sentence_model.encode(segment_text, convert_to_tensor=True)
            similarity = torch.nn.functional.cosine_similarity(
                ad_embedding.unsqueeze(0),
                segment_embedding.unsqueeze(0)
            ).item()
            
            # Keyword matching bonus
            keyword_score = sum(1 for kw in keywords if kw.lower() in segment_text.lower()) / max(len(keywords), 1)
            
            # Combined score
            final_score = (similarity * 0.7) + (keyword_score * 0.3)
            
            scored_segments.append({
                **segment,
                'relevance_score': final_score,
                'semantic_similarity': similarity,
                'keyword_match': keyword_score
            })
        
        # Sort by relevance
        scored_segments.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Find the best insertion point (after the most relevant segment)
        best_segment = scored_segments[0] if scored_segments else None
        
        # Also find natural break points (end of sentences, pauses)
        insertion_candidates = self._find_insertion_points(transcription, scored_segments)
        
        return {
            'segments': scored_segments,
            'best_segment': best_segment,
            'insertion_candidates': insertion_candidates,
            'full_transcription': transcription['full_text'],
            'duration': transcription['duration']
        }
    
    def _create_segments(self, chunks: list, target_duration: float = 30) -> list:
        """Group chunks into larger segments"""
        segments = []
        current_segment = {
            'text': '',
            'start': 0,
            'end': 0,
            'chunks': []
        }
        
        for chunk in chunks:
            if current_segment['text'] == '':
                current_segment['start'] = chunk['start']
            
            current_segment['text'] += ' ' + chunk['text']
            current_segment['end'] = chunk['end']
            current_segment['chunks'].append(chunk)
            
            # Check if segment is long enough
            segment_duration = current_segment['end'] - current_segment['start']
            if segment_duration >= target_duration:
                current_segment['text'] = current_segment['text'].strip()
                segments.append(current_segment)
                current_segment = {
                    'text': '',
                    'start': chunk['end'],
                    'end': chunk['end'],
                    'chunks': []
                }
        
        # Add remaining segment
        if current_segment['text'].strip():
            current_segment['text'] = current_segment['text'].strip()
            segments.append(current_segment)
        
        return segments
    
    def _find_insertion_points(self, transcription: dict, scored_segments: list) -> list:
        """Find natural insertion points after relevant segments"""
        insertion_points = []
        
        # Get top 3 relevant segments
        top_segments = scored_segments[:3]
        
        for segment in top_segments:
            # Find the end of the segment (natural pause point)
            end_time = segment['end']
            
            # Prefer points at sentence boundaries
            text = segment['text']
            
            # Check if segment ends with sentence-ending punctuation
            is_natural_break = text.rstrip().endswith(('.', '!', '?'))
            
            insertion_points.append({
                'timestamp': end_time,
                'relevance_score': segment['relevance_score'],
                'is_natural_break': is_natural_break,
                'context': text[-100:] if len(text) > 100 else text,  # Last 100 chars for context
                'segment_index': scored_segments.index(segment)
            })
        
        return insertion_points
    
    def get_speaker_sample(self, audio_path: str, duration: float = 30) -> str:
        """
        Extract a clean sample of the speaker's voice for cloning
        """
        audio = AudioSegment.from_file(audio_path)
        
        # Skip first 30 seconds (usually intro music/ads)
        start_ms = 30000
        end_ms = start_ms + int(duration * 1000)
        
        # Make sure we don't exceed audio length
        if end_ms > len(audio):
            end_ms = len(audio)
            start_ms = max(0, end_ms - int(duration * 1000))
        
        sample = audio[start_ms:end_ms]
        
        # Save sample
        temp_sample = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sample.export(temp_sample.name, format='wav')
        
        return temp_sample.name


# Test function
if __name__ == "__main__":
    analyzer = PodcastAnalyzer()
    
    # Test with sample ad analysis
    ad_analysis = {
        'problem_solved': 'financial management and budgeting',
        'product_description': 'A budgeting app that helps track expenses',
        'keywords': ['budget', 'money', 'finance', 'save', 'expenses', 'tracking']
    }
    
    print("Podcast Analyzer initialized successfully!")
    print(f"Using device: {analyzer.device}")
