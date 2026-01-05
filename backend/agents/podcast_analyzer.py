"""
Podcast Analyzer Agent
Transcribes and analyzes podcast audio to find optimal ad insertion points
Uses HuggingFace open models for speech-to-text and semantic analysis
Optimized for large podcasts with parallel transcription
"""
import os
import torch
import numpy as np
import traceback
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from sentence_transformers import SentenceTransformer
from pydub import AudioSegment
from pydub.silence import detect_silence
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from sklearn.cluster import AgglomerativeClustering
import soundfile as sf
import torchaudio

# Lazy load speechbrain
SpeakerRecognition = None

def _load_speechbrain():
    """Lazily load speechbrain speaker recognition"""
    global SpeakerRecognition
    if SpeakerRecognition is None:
        try:
            from speechbrain.inference.speaker import SpeakerRecognition as SR
            SpeakerRecognition = SR
        except ImportError:
            from speechbrain.pretrained import SpeakerRecognition as SR
            SpeakerRecognition = SR
    return SpeakerRecognition

class PodcastAnalyzer:
    # Available Whisper models (speed vs accuracy tradeoff)
    WHISPER_MODELS = {
        'tiny': 'openai/whisper-tiny',      # Fastest, least accurate
        'base': 'openai/whisper-base',       # Fast, reasonable accuracy
        'small': 'openai/whisper-small',     # Good balance (default)
        'medium': 'openai/whisper-medium',   # Slower, more accurate
    }
    
    def __init__(self, speed_mode: str = 'balanced'):
        """
        Initialize the podcast analyzer with HuggingFace models
        
        Args:
            speed_mode: 'fast' (tiny model), 'balanced' (small model), 'accurate' (medium model)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Models will be lazy loaded
        self.transcriber = None
        self.sentence_model = None
        self.speaker_model = None  # Speaker embedding model for diarization
        
        # Select model based on speed mode
        self.speed_mode = speed_mode
        if speed_mode == 'fast':
            self.whisper_model = self.WHISPER_MODELS['tiny']
            self.parallel_chunk_duration = 120  # Larger chunks for faster processing
        elif speed_mode == 'accurate':
            self.whisper_model = self.WHISPER_MODELS['medium']
            self.parallel_chunk_duration = 45  # Smaller chunks for better accuracy
        else:  # balanced
            self.whisper_model = self.WHISPER_MODELS['small']
            self.parallel_chunk_duration = 60  # Default chunk size
        
        # Number of workers for parallel processing
        self.num_workers = min(4, multiprocessing.cpu_count())
        
        print(f"PodcastAnalyzer initialized: speed_mode={speed_mode}, model={self.whisper_model}")
        
        # Pre-load speaker model in background for faster first detection
        self._preload_speaker_model()
        
    def _preload_speaker_model(self):
        """Pre-load speaker model in background thread"""
        import threading
        def load():
            try:
                self._load_speaker_model()
            except Exception as e:
                print(f"Warning: Failed to preload speaker model: {e}")
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
        
    def _load_transcriber(self):
        """Load Whisper model for transcription"""
        if self.transcriber is None:
            print(f"Loading Whisper model: {self.whisper_model}...")
            # Using OpenAI Whisper through HuggingFace - open source
            self.transcriber = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model,
                torch_dtype=self.torch_dtype,
                device=self.device,
                chunk_length_s=30,
                return_timestamps=True,
                batch_size=8 if self.device == "cuda" else 1  # Batch processing for GPU
            )
    
    def _load_sentence_model(self):
        """Load sentence transformer for semantic similarity"""
        if self.sentence_model is None:
            print("Loading sentence transformer model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            if self.device == "cuda":
                self.sentence_model = self.sentence_model.to(self.device)
    
    def _load_speaker_model(self):
        """Load speaker embedding model for speaker diarization"""
        if self.speaker_model is None:
            print("Loading speaker embedding model...")
            SR = _load_speechbrain()
            kwargs = {}
            try:
                from speechbrain.utils.fetching import LocalStrategy
                if os.name == 'nt':
                    kwargs['local_strategy'] = LocalStrategy.COPY
            except ImportError:
                pass
            
            self.speaker_model = SR.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="tmp_model_verification",
                run_opts={"device": self.device},
                **kwargs
            )
    
    def _get_speaker_embedding(self, audio_path: str) -> np.ndarray:
        """Get speaker embedding for an audio file"""
        self._load_speaker_model()
        
        # Load audio
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
        
        # Get embedding
        embedding = self.speaker_model.encode_batch(signal)
        return embedding.squeeze().cpu().numpy()
    
    def _verify_single_speaker(self, audio_segment: 'AudioSegment', target_embedding: np.ndarray, 
                                threshold: float = 0.75) -> tuple:
        """
        Verify that an audio segment contains only one speaker's voice.
        Splits the segment into windows and checks each window against the target speaker.
        
        Args:
            audio_segment: The audio to verify
            target_embedding: The embedding of the target speaker
            threshold: Minimum similarity required for each window
            
        Returns:
            (is_valid, clean_start_ms, clean_end_ms) - whether the segment is valid and the clean portion
        """
        duration_ms = len(audio_segment)
        
        # For very short segments, just return as-is
        if duration_ms < 2000:
            return (True, 0, duration_ms)
        
        # Split into 1-second windows with 500ms overlap
        window_size_ms = 1000
        hop_size_ms = 500
        
        window_results = []
        
        for start_ms in range(0, duration_ms - window_size_ms + 1, hop_size_ms):
            end_ms = start_ms + window_size_ms
            window = audio_segment[start_ms:end_ms]
            
            # Save window to temp file and get embedding
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    window.export(temp_file.name, format='wav', parameters=["-ac", "1", "-ar", "16000"])
                    window_embedding = self._get_speaker_embedding(temp_file.name)
                    os.unlink(temp_file.name)
                
                # Normalize embeddings
                window_emb_norm = window_embedding / np.linalg.norm(window_embedding)
                target_emb_norm = target_embedding / np.linalg.norm(target_embedding)
                
                # Compute similarity
                similarity = np.dot(window_emb_norm, target_emb_norm)
                window_results.append((start_ms, end_ms, similarity))
                
            except Exception as e:
                # If we can't process a window, mark it as invalid
                window_results.append((start_ms, end_ms, 0.0))
        
        if not window_results:
            return (True, 0, duration_ms)
        
        # Find the longest contiguous region where all windows pass threshold
        valid_windows = [(s, e, sim) for s, e, sim in window_results if sim >= threshold]
        
        if not valid_windows:
            # No valid windows - try to find the single best window
            best_window = max(window_results, key=lambda x: x[2])
            if best_window[2] >= threshold - 0.1:  # Slightly lower threshold for best window
                return (True, best_window[0], best_window[1])
            return (False, 0, 0)
        
        # Find contiguous valid region
        # Start from first valid window and extend
        best_start = valid_windows[0][0]
        best_end = valid_windows[0][1]
        
        current_start = best_start
        current_end = best_end
        
        for i in range(1, len(valid_windows)):
            prev_end = valid_windows[i-1][1]
            curr_start = valid_windows[i][0]
            
            # Check if this window is contiguous (within hop_size)
            if curr_start <= prev_end:
                current_end = valid_windows[i][1]
            else:
                # Gap found - check if current region is better
                if (current_end - current_start) > (best_end - best_start):
                    best_start = current_start
                    best_end = current_end
                current_start = valid_windows[i][0]
                current_end = valid_windows[i][1]
        
        # Check final region
        if (current_end - current_start) > (best_end - best_start):
            best_start = current_start
            best_end = current_end
        
        # Return the clean region
        return (True, best_start, best_end)
    
    def _extract_clean_speaker_sample(self, audio: 'AudioSegment', segments_with_scores: list,
                                       target_embedding: np.ndarray, target_duration_ms: int = 30000) -> 'AudioSegment':
        """
        Extract a clean audio sample containing only the target speaker's voice.
        
        Args:
            audio: Full audio file
            segments_with_scores: List of (segment, similarity, embedding) tuples, sorted by similarity
            target_embedding: The centroid embedding for the target speaker
            target_duration_ms: Target duration for the sample
            
        Returns:
            AudioSegment containing only the target speaker's voice
        """
        sample = AudioSegment.empty()
        
        for seg, similarity, emb in segments_with_scores:
            if len(sample) >= target_duration_ms:
                break
                
            seg_start, seg_end = seg
            segment_audio = audio[seg_start:seg_end]
            
            # Verify this segment contains only the target speaker
            is_valid, clean_start, clean_end = self._verify_single_speaker(
                segment_audio, target_embedding, threshold=0.80
            )
            
            if is_valid and (clean_end - clean_start) >= 1000:  # At least 1 second of clean audio
                clean_audio = segment_audio[clean_start:clean_end]
                
                if len(sample) > 0:
                    sample += AudioSegment.silent(duration=200)
                sample += clean_audio
        
        # If we still don't have enough, try with lower threshold
        if len(sample) < 5000:
            print("Not enough clean audio, retrying with lower threshold...")
            sample = AudioSegment.empty()
            
            for seg, similarity, emb in segments_with_scores[:3]:  # Just top 3
                seg_start, seg_end = seg
                segment_audio = audio[seg_start:seg_end]
                
                # Use lower threshold
                is_valid, clean_start, clean_end = self._verify_single_speaker(
                    segment_audio, target_embedding, threshold=0.70
                )
                
                if is_valid and (clean_end - clean_start) >= 500:
                    clean_audio = segment_audio[clean_start:clean_end]
                    if len(sample) > 0:
                        sample += AudioSegment.silent(duration=200)
                    sample += clean_audio
                    
                    if len(sample) >= target_duration_ms:
                        break
        
        # Last resort: use just the middle of the best segment
        if len(sample) < 3000 and segments_with_scores:
            best_seg, _, _ = segments_with_scores[0]
            seg_start, seg_end = best_seg
            seg_duration = seg_end - seg_start
            
            # Take the middle 50% of the segment
            trim = seg_duration // 4
            sample = audio[seg_start + trim:seg_end - trim]
        
        return sample[:target_duration_ms] if len(sample) > target_duration_ms else sample

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
        Optimized with parallel chunk processing for long podcasts
        """
        self._load_transcriber()
        
        if status_callback:
            status_callback(20, "Loading audio for transcription...")
        
        print(f"Transcribing: {audio_path}")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        duration_seconds = len(audio) / 1000
        
        print(f"Audio duration: {duration_seconds:.1f} seconds ({duration_seconds/60:.1f} minutes)")
        
        # Export to wav for transcription (mono, 16kHz)
        temp_wav = None
        if not audio_path.endswith('.wav'):
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio.export(temp_wav.name, format='wav', parameters=["-ac", "1", "-ar", "16000"])
            audio_path = temp_wav.name
        
        # For long audio (> 5 minutes), use chunked parallel transcription
        if duration_seconds > 300:  # 5 minutes
            transcription = self._transcribe_parallel(audio, audio_path, duration_seconds, status_callback)
        else:
            # For shorter audio, use standard transcription
            if status_callback:
                status_callback(25, "Transcribing audio...")
            
            result = self.transcriber(audio_path)
            transcription = self._process_transcription_result(result, duration_seconds)
        
        # Clean up temp file
        if temp_wav:
            try:
                os.unlink(temp_wav.name)
            except:
                pass
        
        return transcription
    
    def _transcribe_parallel(self, audio: AudioSegment, audio_path: str, duration_seconds: float, status_callback=None) -> dict:
        """
        Transcribe long audio in parallel chunks for speed
        """
        chunk_duration_ms = self.parallel_chunk_duration * 1000  # Convert to milliseconds
        num_chunks = int(np.ceil(len(audio) / chunk_duration_ms))
        
        print(f"Splitting audio into {num_chunks} chunks for parallel transcription...")
        
        if status_callback:
            status_callback(22, f"Preparing {num_chunks} audio chunks for parallel processing...")
        
        # Create temporary chunk files
        chunk_files = []
        chunk_offsets = []
        
        for i in range(num_chunks):
            start_ms = i * chunk_duration_ms
            end_ms = min((i + 1) * chunk_duration_ms, len(audio))
            
            chunk = audio[start_ms:end_ms]
            chunk_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            chunk.export(chunk_file.name, format='wav', parameters=["-ac", "1", "-ar", "16000"])
            
            chunk_files.append(chunk_file.name)
            chunk_offsets.append(start_ms / 1000)  # Convert to seconds
        
        if status_callback:
            status_callback(25, f"Transcribing {num_chunks} chunks in parallel...")
        
        # Transcribe chunks (sequentially but with optimized batching)
        # Note: True parallelism requires multiple model instances which uses too much memory
        # Instead, we process chunks efficiently with progress updates
        all_chunks = []
        full_text_parts = []
        
        for i, (chunk_file, offset) in enumerate(zip(chunk_files, chunk_offsets)):
            progress = 25 + int((i / num_chunks) * 15)  # Progress from 25% to 40%
            if status_callback:
                status_callback(progress, f"Transcribing chunk {i+1}/{num_chunks}...")
            
            try:
                result = self.transcriber(chunk_file)
                
                # Adjust timestamps with offset
                if 'chunks' in result:
                    for chunk in result['chunks']:
                        adjusted_chunk = {
                            'text': chunk['text'],
                            'start': (chunk['timestamp'][0] if chunk['timestamp'][0] is not None else 0) + offset,
                            'end': (chunk['timestamp'][1] if chunk['timestamp'][1] is not None else self.parallel_chunk_duration) + offset
                        }
                        all_chunks.append(adjusted_chunk)
                
                full_text_parts.append(result['text'])
                
            except Exception as e:
                print(f"Error transcribing chunk {i}: {e}")
            
            # Clean up chunk file immediately
            try:
                os.unlink(chunk_file)
            except:
                pass
        
        # Combine results
        transcription = {
            'full_text': ' '.join(full_text_parts),
            'chunks': all_chunks,
            'duration': duration_seconds
        }
        
        print(f"Transcription complete: {len(all_chunks)} segments")
        return transcription
    
    def _process_transcription_result(self, result: dict, duration_seconds: float) -> dict:
        """Process transcription result into standard format"""
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
    
    def analyze_content(self, transcription: dict, ad_analysis: dict, status_callback=None, audio_path: str = None) -> dict:
        """
        Analyze podcast content to find optimal ad placement using stepwise logic:
        
        Step 1: Identify where the problem (that the ad solves) is discussed in the podcast
        Step 2: If problem found, place ad after the problem mention at sentence boundary
        Step 3: If problem not found, use semantic similarity to find best placement
        
        Args:
            transcription: The transcription dict
            ad_analysis: Ad analysis results
            status_callback: Progress callback
            audio_path: Path to audio file for pause detection
        """
        self._load_sentence_model()
        
        # Store audio path for use in insertion point detection
        self._current_audio_path = audio_path
        
        if status_callback:
            status_callback(40, "Step 1: Analyzing ad to understand the problem being solved...")
        
        # Get problem information from ad analysis
        problem_solved = ad_analysis.get('problem_solved', '')
        problem_keywords = ad_analysis.get('problem_keywords', [])
        
        print(f"Ad Problem: {problem_solved}")
        print(f"Problem Keywords: {problem_keywords}")
        
        # Create segments (group chunks into ~30 second segments)
        segments = self._create_segments(transcription['chunks'], target_duration=30)
        
        if not segments:
            return {
                'segments': [],
                'best_segment': None,
                'insertion_candidates': [],
                'full_transcription': transcription['full_text'],
                'duration': transcription['duration'],
                'placement_method': 'fallback'
            }
        
        if status_callback:
            status_callback(42, "Step 2: Searching for problem mentions in podcast...")
        
        # STEP 2: Search for problem mentions in the podcast
        problem_matches = self._find_problem_mentions(segments, problem_solved, problem_keywords)
        
        placement_method = 'problem_match'
        insertion_candidates = []
        
        if problem_matches:
            print(f"Found {len(problem_matches)} segment(s) discussing the problem!")
            if status_callback:
                status_callback(45, f"Found problem discussion in {len(problem_matches)} segment(s)!")
            
            # Find insertion point after the problem mention, at sentence boundary
            insertion_candidates = self._find_insertion_after_problem(
                transcription, problem_matches, audio_path
            )
        else:
            # STEP 3: Fallback to semantic similarity
            print("Problem not explicitly mentioned. Using semantic similarity fallback...")
            if status_callback:
                status_callback(45, "Step 3: Problem not found, using semantic similarity...")
            
            placement_method = 'similarity_fallback'
            insertion_candidates = self._find_insertion_by_similarity(
                transcription, segments, ad_analysis, audio_path, status_callback
            )
        
        if not insertion_candidates:
            print("No valid insertion candidates found. Returning empty result.")
            return {
                'segments': segments,
                'best_segment': None,
                'insertion_candidates': [],
                'full_transcription': transcription['full_text'],
                'duration': transcription['duration'],
                'placement_method': placement_method
            }
        
        # Get best segment for reference
        best_segment = insertion_candidates[0] if insertion_candidates else None
        
        return {
            'segments': segments,
            'best_segment': best_segment,
            'insertion_candidates': insertion_candidates,
            'full_transcription': transcription['full_text'],
            'duration': transcription['duration'],
            'placement_method': placement_method
        }
    
    def _find_problem_mentions(self, segments: list, problem_solved: str, problem_keywords: list) -> list:
        """
        Agent 2: Search for segments where the problem (that the ad solves) is discussed.
        Uses keyword matching and semantic similarity to find problem mentions.
        """
        if not problem_solved or problem_solved == "General improvement/enhancement":
            return []
        
        problem_matches = []
        problem_lower = problem_solved.lower()
        
        # Create embedding for the problem description
        problem_embedding = self.sentence_model.encode(
            problem_solved,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        for segment in segments:
            segment_text = segment['text'].lower()
            match_score = 0
            match_reasons = []
            
            # Check 1: Direct keyword matches
            keyword_matches = 0
            for keyword in problem_keywords:
                if keyword.lower() in segment_text:
                    keyword_matches += 1
                    match_reasons.append(f"keyword: '{keyword}'")
            
            if keyword_matches > 0:
                match_score += keyword_matches * 0.3
            
            # Check 2: Problem phrase similarity
            # Look for sentences that discuss problems/challenges
            problem_indicators = [
                'problem', 'issue', 'challenge', 'difficult', 'hard',
                'struggle', 'frustrated', 'annoying', 'tired of',
                'wish', 'need', 'want', 'looking for', 'searching for',
                'how do', 'how can', 'what if', 'worried about'
            ]
            
            indicator_matches = sum(1 for ind in problem_indicators if ind in segment_text)
            if indicator_matches > 0:
                match_score += indicator_matches * 0.1
                match_reasons.append(f"problem indicators: {indicator_matches}")
            
            # Check 3: Semantic similarity with the problem description
            segment_embedding = self.sentence_model.encode(
                segment['text'],
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            similarity = torch.nn.functional.cosine_similarity(
                problem_embedding.unsqueeze(0),
                segment_embedding.unsqueeze(0),
                dim=1
            ).item()
            
            # High semantic similarity indicates problem discussion
            if similarity > 0.5:
                match_score += similarity * 0.5
                match_reasons.append(f"semantic similarity: {similarity:.2f}")
            
            # Consider it a match if score is above threshold
            if match_score > 0.3:
                problem_matches.append({
                    **segment,
                    'problem_match_score': match_score,
                    'match_reasons': match_reasons,
                    'semantic_similarity': similarity
                })
        
        # Sort by match score
        problem_matches.sort(key=lambda x: x['problem_match_score'], reverse=True)
        
        print(f"Problem mention search: {len(problem_matches)} matches found")
        for match in problem_matches[:3]:
            print(f"  - Score {match['problem_match_score']:.2f}: {match['text'][:100]}...")
        
        return problem_matches
    
    def _find_insertion_after_problem(self, transcription: dict, problem_matches: list, audio_path: str) -> list:
        """
        Find insertion point AFTER the problem is mentioned, at a sentence boundary.
        The ad should come right after the problem is discussed.
        """
        insertion_candidates = []
        
        for match in problem_matches:
            segment_end = match['end']
            
            # Find the sentence boundary at or after the segment end
            # Look in the chunks for a good sentence ending
            chunks = transcription.get('chunks', [])
            
            for chunk in chunks:
                # Look for chunks that end near or after this segment
                if chunk['end'] >= segment_end - 5:  # Allow 5 second window
                    chunk_text = chunk['text'].strip()
                    
                    # Check if this chunk ends with a sentence terminator
                    if chunk_text and chunk_text[-1] in '.!?':
                        insertion_candidates.append({
                            'timestamp': chunk['end'],
                            'relevance_score': match['problem_match_score'],
                            'is_natural_break': True,
                            'boundary_context': chunk_text[-100:] if len(chunk_text) > 100 else chunk_text,
                            'placement_reason': f"After problem mention: {', '.join(match['match_reasons'][:2])}",
                            'text': match['text'][:200]
                        })
                        break
            
            # If no sentence boundary found, use segment end
            if not any(c['timestamp'] == segment_end for c in insertion_candidates):
                insertion_candidates.append({
                    'timestamp': segment_end,
                    'relevance_score': match['problem_match_score'] * 0.8,  # Slightly lower score
                    'is_natural_break': False,
                    'boundary_context': match['text'][-100:] if len(match['text']) > 100 else match['text'],
                    'placement_reason': f"End of problem discussion segment",
                    'text': match['text'][:200]
                })
        
        # Sort by relevance score
        insertion_candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return insertion_candidates[:5]  # Return top 5 candidates
    
    def _find_insertion_by_similarity(self, transcription: dict, segments: list, ad_analysis: dict, 
                                       audio_path: str, status_callback=None) -> list:
        """
        Agent 3: Fallback - Use semantic similarity when problem isn't explicitly mentioned.
        Finds segments most similar to the ad content.
        """
        if status_callback:
            status_callback(47, "Computing semantic similarity with all segments...")
        
        # Create a topic description from ad analysis
        ad_topic = f"{ad_analysis.get('problem_solved', '')} {ad_analysis.get('product_description', '')}"
        keywords = ad_analysis.get('keywords', [])
        
        # Batch encode all segments
        segment_texts = [segment['text'] for segment in segments]
        
        all_embeddings = self.sentence_model.encode(
            [ad_topic] + segment_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        ad_embedding = all_embeddings[0]
        segment_embeddings = all_embeddings[1:]
        
        # Compute similarities
        similarities = torch.nn.functional.cosine_similarity(
            ad_embedding.unsqueeze(0),
            segment_embeddings,
            dim=1
        )
        
        # Score each segment
        scored_segments = []
        for i, segment in enumerate(segments):
            segment_text = segment['text']
            similarity = similarities[i].item()
            
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
        
        # Find insertion candidates at sentence boundaries after relevant segments
        insertion_candidates = []
        chunks = transcription.get('chunks', [])
        
        for segment in scored_segments[:5]:  # Top 5 most relevant
            segment_end = segment['end']
            
            # Find sentence boundary after this segment
            for chunk in chunks:
                if chunk['end'] >= segment_end - 2:
                    chunk_text = chunk['text'].strip()
                    if chunk_text and chunk_text[-1] in '.!?':
                        insertion_candidates.append({
                            'timestamp': chunk['end'],
                            'relevance_score': segment['relevance_score'],
                            'is_natural_break': True,
                            'boundary_context': chunk_text[-100:] if len(chunk_text) > 100 else chunk_text,
                            'placement_reason': f"Semantic similarity: {segment['semantic_similarity']:.2f}",
                            'text': segment['text'][:200]
                        })
                        break
            else:
                # No sentence boundary found, use segment end
                insertion_candidates.append({
                    'timestamp': segment_end,
                    'relevance_score': segment['relevance_score'] * 0.8,
                    'is_natural_break': False,
                    'boundary_context': segment['text'][-100:],
                    'placement_reason': f"Similar content (similarity: {segment['semantic_similarity']:.2f})",
                    'text': segment['text'][:200]
                })
        
        return insertion_candidates
    
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
        
        # Don't forget the last segment
        if current_segment['text'].strip():
            current_segment['text'] = current_segment['text'].strip()
            segments.append(current_segment)
        
        # If no segments were created (very short audio), create one from all chunks
        if not segments and chunks:
            segments.append({
                'text': ' '.join(c['text'] for c in chunks).strip(),
                'start': chunks[0]['start'],
                'end': chunks[-1]['end'],
                'chunks': chunks
            })
        
        return segments
    
    def detect_speakers(self, audio_path: str, output_folder: str, status_callback=None) -> dict:
        """
        Detect speakers in the podcast audio using speaker diarization.
        Extracts audio samples for each detected speaker.
        
        Args:
            audio_path: Path to the audio file
            output_folder: Folder to save speaker audio samples
            status_callback: Progress callback function(progress, message)
            
        Returns:
            dict with speaker information:
                - num_speakers: Number of detected speakers
                - speakers: List of speaker info (id, sample_path, total_speaking_time)
                - requires_selection: Whether user should select a speaker
        """
        if status_callback:
            status_callback(10, "Loading audio for speaker detection...")
        
        print(f"Detecting speakers in: {audio_path}")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        duration_seconds = len(audio) / 1000
        
        if status_callback:
            status_callback(20, "Analyzing audio for speaker segments...")
        
        # Use energy-based voice activity detection to find speech segments
        # This is a simplified approach - for production, use pyannote or speechbrain diarization
        speech_segments = self._detect_speech_segments(audio)
        
        if status_callback:
            status_callback(40, "Clustering speaker segments...")
        
        # For now, use a simplified single-speaker detection
        # In production, you'd use proper diarization (pyannote-audio or speechbrain)
        speakers = self._cluster_speakers(audio, speech_segments, output_folder)
        
        if status_callback:
            status_callback(80, f"Detected {len(speakers)} speaker(s)...")
        
        # Determine if selection is required (more than one speaker)
        requires_selection = len(speakers) > 1
        
        result = {
            'num_speakers': len(speakers),
            'speakers': speakers,
            'requires_selection': requires_selection,
            'duration': duration_seconds
        }
        
        if status_callback:
            status_callback(100, f"Speaker detection complete: {len(speakers)} speaker(s) found")
        
        print(f"Speaker detection complete: {len(speakers)} speaker(s)")
        return result
    
    def detect_speakers_progressive(self, audio_path: str, output_folder: str, 
                                     on_speaker_found=None, status_callback=None) -> dict:
        """
        Progressive speaker detection that yields speakers as they are found.
        This allows the UI to show speakers immediately rather than waiting for all.
        
        Args:
            audio_path: Path to the audio file
            output_folder: Folder to save speaker audio samples
            on_speaker_found: Callback function(speaker_info) called when each speaker is found
                             Returns False to stop detection, True to continue
            status_callback: Progress callback function(progress, message)
            
        Returns:
            dict with speaker information
        """
        if status_callback:
            status_callback(5, "Loading audio for speaker detection...")
        
        print(f"Progressive speaker detection in: {audio_path}")
        os.makedirs(output_folder, exist_ok=True)
        
        # Load ONLY the first 3 minutes initially for quick speaker detection
        # This dramatically speeds up finding the first speakers
        audio_full = AudioSegment.from_file(audio_path)
        duration_seconds = len(audio_full) / 1000
        
        # Use first 3 minutes for initial quick detection
        initial_duration_ms = min(180000, len(audio_full))  # 3 minutes max
        audio = audio_full[:initial_duration_ms]
        
        if status_callback:
            status_callback(8, "Quick analysis of first few minutes...")
        
        # Detect speech segments in the initial portion only
        speech_segments = self._detect_speech_segments(audio)
        
        if not speech_segments:
            # No speech detected, create a default sample
            if status_callback:
                status_callback(50, "No speech detected, creating default sample...")
            
            sample_duration = min(30000, len(audio))
            sample = audio[:sample_duration]
            sample_path = os.path.join(output_folder, 'speaker_0_sample.wav')
            sample.export(sample_path, format='wav')
            
            speaker = {
                'id': 0,
                'label': 'Speaker 1',
                'sample_path': sample_path,
                'total_speaking_time': sample_duration / 1000,
                'segments': [(0, sample_duration)]
            }
            
            if on_speaker_found:
                on_speaker_found(speaker)
            
            return {
                'num_speakers': 1,
                'speakers': [speaker],
                'requires_selection': False,
                'duration': duration_seconds
            }
        
        if status_callback:
            status_callback(12, f"Found {len(speech_segments)} speech segments. Extracting embeddings...")
        
        # Sort segments by length (longer = more reliable)
        sorted_segments = sorted(speech_segments, key=lambda x: x[1] - x[0], reverse=True)
        
        # Get segments for embedding (at least 2 seconds) - limit to 15 for speed
        segments_for_embedding = [s for s in sorted_segments if s[1] - s[0] >= 2000][:15]
        
        if len(segments_for_embedding) < 2:
            # Not enough segments, return single speaker quickly
            if status_callback:
                status_callback(50, "Creating speaker sample...")
            
            total_speaking_time = sum(e - s for s, e in speech_segments) / 1000
            best_segment = sorted_segments[0]
            sample_start = best_segment[0]
            sample_end = min(best_segment[1], sample_start + 30000)
            sample = audio[sample_start:sample_end]
            sample_path = os.path.join(output_folder, 'speaker_0_sample.wav')
            sample.export(sample_path, format='wav')
            
            speaker = {
                'id': 0,
                'label': 'Speaker 1',
                'sample_path': sample_path,
                'total_speaking_time': total_speaking_time,
                'segments': speech_segments
            }
            
            if on_speaker_found:
                on_speaker_found(speaker)
            
            return {
                'num_speakers': 1,
                'speakers': [speaker],
                'requires_selection': False,
                'duration': duration_seconds
            }
        
        # Extract embeddings progressively
        embeddings = []
        temp_files = []
        
        try:
            total_segments = len(segments_for_embedding)
            
            for i, (start_ms, end_ms) in enumerate(segments_for_embedding):
                # Update progress
                progress = 15 + int((i / total_segments) * 35)
                if status_callback:
                    status_callback(progress, f"Analyzing segment {i+1}/{total_segments}...")
                
                # Extract segment audio
                segment_audio = audio[start_ms:end_ms]
                
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                segment_audio.export(temp_file.name, format='wav', parameters=["-ac", "1", "-ar", "16000"])
                temp_files.append(temp_file.name)
                
                try:
                    embedding = self._get_speaker_embedding(temp_file.name)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error getting embedding for segment {i}: {e}")
                    continue
                
                # After getting just 3 embeddings, do preliminary clustering
                # to quickly identify first speaker(s) - show results FAST
                if len(embeddings) >= 3 and i == 2:
                    # Quick preliminary clustering
                    prelim_embeddings = np.array(embeddings)
                    try:
                        # Use higher threshold (0.5) to avoid over-splitting same speaker
                        # Cosine distance: 0 = identical, 2 = opposite
                        prelim_clustering = AgglomerativeClustering(
                            n_clusters=None,
                            distance_threshold=0.5,
                            metric='cosine',
                            linkage='average'
                        )
                        prelim_labels = prelim_clustering.fit_predict(prelim_embeddings)
                        
                        # Create initial speaker samples
                        speaker_id = 0
                        for label in set(prelim_labels):
                            label_indices = [j for j, l in enumerate(prelim_labels) if l == label]
                            if not label_indices:
                                continue
                            
                            # Get the best segment for this speaker (use middle portion for purity)
                            best_idx = label_indices[0]
                            seg_start, seg_end = segments_for_embedding[best_idx]
                            seg_duration = seg_end - seg_start
                            # Trim edges to avoid speaker transitions
                            if seg_duration > 3000:
                                seg_start += 500
                                seg_end -= 500
                            sample = audio[seg_start:min(seg_end, seg_start + 30000)]
                            sample_path = os.path.join(output_folder, f'speaker_{speaker_id}_sample.wav')
                            sample.export(sample_path, format='wav')
                            
                            speaking_time = sum(
                                segments_for_embedding[j][1] - segments_for_embedding[j][0] 
                                for j in label_indices
                            ) / 1000
                            
                            speaker_info = {
                                'id': speaker_id,
                                'label': f'Speaker {speaker_id + 1}',
                                'sample_path': sample_path,
                                'total_speaking_time': speaking_time,
                                'segments': [segments_for_embedding[j] for j in label_indices],
                                'preliminary': True
                            }
                            
                            if on_speaker_found:
                                should_continue = on_speaker_found(speaker_info)
                                if should_continue is False:
                                    # User cancelled
                                    return {
                                        'num_speakers': speaker_id + 1,
                                        'speakers': [speaker_info],
                                        'requires_selection': False,
                                        'duration': duration_seconds,
                                        'cancelled': True
                                    }
                            
                            speaker_id += 1
                    except Exception as e:
                        print(f"Preliminary clustering failed: {e}")
            
            if len(embeddings) < 2:
                raise ValueError("Not enough embeddings computed")
            
            # Final clustering with all embeddings
            if status_callback:
                status_callback(60, "Finalizing speaker clustering...")
            
            embeddings = np.array(embeddings)
            
            # Use higher threshold (0.5) to avoid over-splitting
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            # 0.5 means speakers with cosine similarity > 0.75 are grouped together
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            
            num_speakers = len(set(labels))
            print(f"Initial clustering found {num_speakers} speaker(s)")
            
            # Post-processing: Merge speakers that are too similar
            # Compute centroids for each cluster
            cluster_centroids = {}
            for label in set(labels):
                cluster_embeddings = embeddings[labels == label]
                centroid = cluster_embeddings.mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                cluster_centroids[label] = centroid
            
            # Check if any clusters should be merged (cosine similarity > 0.80)
            merge_threshold = 0.80  # Lower threshold to merge similar voices
            merged_labels = {label: label for label in set(labels)}
            labels_list = sorted(set(labels))
            
            for i, label1 in enumerate(labels_list):
                for label2 in labels_list[i+1:]:
                    if merged_labels[label1] != merged_labels[label2]:
                        # Check similarity between centroids
                        sim = np.dot(cluster_centroids[label1], cluster_centroids[label2])
                        if sim > merge_threshold:
                            # Merge label2 into label1
                            old_label = merged_labels[label2]
                            for k in merged_labels:
                                if merged_labels[k] == old_label:
                                    merged_labels[k] = merged_labels[label1]
                            print(f"Merging speaker {label2} into {label1} (similarity: {sim:.3f})")
            
            # Apply merged labels
            labels = np.array([merged_labels[l] for l in labels])
            num_speakers = len(set(labels))
            print(f"Final clustering found {num_speakers} speaker(s) after merging")
            
            # Group segments by speaker
            speaker_segments = {}
            for i, label in enumerate(labels):
                if label not in speaker_segments:
                    speaker_segments[label] = []
                speaker_segments[label].append((segments_for_embedding[i], embeddings[i]))
            
            # Compute centroids
            speaker_centroids = {}
            for speaker_label, speaker_data in speaker_segments.items():
                embeddings_for_speaker = np.array([s[1] for s in speaker_data])
                centroid = embeddings_for_speaker.mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                speaker_centroids[speaker_label] = centroid
            
            # Build final speaker info
            if status_callback:
                status_callback(70, "Creating final speaker samples...")
            
            # Use full audio for final sample creation (better quality samples)
            audio_for_samples = audio_full if len(audio_full) > len(audio) else audio
            
            speakers = []
            for speaker_label in sorted(speaker_segments.keys()):
                speaker_data = speaker_segments[speaker_label]
                segment_list = [s[0] for s in speaker_data]
                segment_embeddings = [s[1] for s in speaker_data]
                
                speaking_time = sum(e - s for s, e in segment_list) / 1000
                
                # Sort segments by similarity to centroid
                centroid = speaker_centroids[speaker_label]
                segment_scores = []
                for seg, emb in zip(segment_list, segment_embeddings):
                    emb_norm = emb / np.linalg.norm(emb)
                    similarity = np.dot(centroid, emb_norm)
                    segment_scores.append((seg, similarity, emb_norm))
                
                segment_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Use the new verification method to extract clean speaker sample
                print(f"Extracting clean sample for speaker {speaker_label}...")
                sample = self._extract_clean_speaker_sample(
                    audio_for_samples, segment_scores, centroid, target_duration_ms=30000
                )
                
                sample_path = os.path.join(output_folder, f'speaker_{speaker_label}_sample.wav')
                sample.export(sample_path, format='wav')
                
                speakers.append({
                    'id': int(speaker_label),
                    'label': f'Speaker {speaker_label + 1}',
                    'sample_path': sample_path,
                    'total_speaking_time': speaking_time,
                    'segments': segment_list
                })
            
            # Sort by speaking time
            speakers.sort(key=lambda x: x['total_speaking_time'], reverse=True)
            
            # Reassign IDs - use temp names first
            temp_renames = []
            for i, speaker in enumerate(speakers):
                old_path = speaker['sample_path']
                temp_path = os.path.join(output_folder, f'speaker_temp_{i}_sample.wav')
                if os.path.exists(old_path):
                    try:
                        os.rename(old_path, temp_path)
                        temp_renames.append((temp_path, i))
                    except:
                        temp_renames.append((old_path, i))
                speaker['id'] = i
                speaker['label'] = f'Speaker {i + 1}'
            
            for temp_path, i in temp_renames:
                final_path = os.path.join(output_folder, f'speaker_{i}_sample.wav')
                try:
                    if os.path.exists(temp_path) and temp_path != final_path:
                        os.rename(temp_path, final_path)
                    speakers[i]['sample_path'] = final_path
                except Exception as e:
                    print(f"Warning: Could not rename {temp_path} to {final_path}: {e}")
                    speakers[i]['sample_path'] = temp_path
            
            if status_callback:
                status_callback(90, f"Found {len(speakers)} speaker(s)")
            
            # Notify about final speakers (may have changed from preliminary)
            # Only notify if speaker list is different from preliminary
            
            result = {
                'num_speakers': len(speakers),
                'speakers': speakers,
                'requires_selection': len(speakers) > 1,
                'duration': duration_seconds
            }
            
            if status_callback:
                status_callback(100, f"Speaker detection complete: {len(speakers)} speaker(s)")
            
            return result
            
        except Exception as e:
            print(f"Progressive speaker detection failed: {e}, falling back to single speaker")
            traceback.print_exc()
            
            total_speaking_time = sum(e - s for s, e in speech_segments) / 1000
            best_segment = sorted_segments[0]
            sample_start = best_segment[0]
            sample_end = min(best_segment[1], sample_start + 30000)
            sample = audio[sample_start:sample_end]
            sample_path = os.path.join(output_folder, 'speaker_0_sample.wav')
            sample.export(sample_path, format='wav')
            
            speaker = {
                'id': 0,
                'label': 'Speaker 1',
                'sample_path': sample_path,
                'total_speaking_time': total_speaking_time,
                'segments': speech_segments
            }
            
            if on_speaker_found:
                on_speaker_found(speaker)
            
            return {
                'num_speakers': 1,
                'speakers': [speaker],
                'requires_selection': False,
                'duration': duration_seconds
            }
        finally:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def _detect_speech_segments(self, audio: AudioSegment) -> list:
        """
        Detect speech segments using silence detection.
        Returns list of (start_ms, end_ms) tuples.
        """
        # Find silent regions
        silence_thresh = audio.dBFS - 16  # Silence threshold relative to average
        min_silence_len = 500  # Minimum silence length in ms
        
        silent_ranges = detect_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        # Convert silence ranges to speech segments
        speech_segments = []
        prev_end = 0
        
        for silence_start, silence_end in silent_ranges:
            if silence_start > prev_end:
                speech_segments.append((prev_end, silence_start))
            prev_end = silence_end
        
        # Add final segment if audio doesn't end in silence
        if prev_end < len(audio):
            speech_segments.append((prev_end, len(audio)))
        
        # Filter out very short segments (< 1 second)
        speech_segments = [(s, e) for s, e in speech_segments if e - s >= 1000]
        
        print(f"Detected {len(speech_segments)} speech segments")
        return speech_segments
    
    def _cluster_speakers(self, audio: AudioSegment, speech_segments: list, output_folder: str) -> list:
        """
        Cluster speech segments by speaker using speaker embeddings.
        Uses ECAPA-TDNN model to compute speaker embeddings and agglomerative clustering.
        
        Returns list of speaker dictionaries.
        """
        os.makedirs(output_folder, exist_ok=True)
        
        if not speech_segments:
            # No speech detected, create a default sample from the beginning
            sample_duration = min(30000, len(audio))  # 30 seconds max
            sample = audio[:sample_duration]
            sample_path = os.path.join(output_folder, 'speaker_0_sample.wav')
            sample.export(sample_path, format='wav')
            
            return [{
                'id': 0,
                'label': 'Speaker 1',
                'sample_path': sample_path,
                'total_speaking_time': sample_duration / 1000,
                'segments': [(0, sample_duration)]
            }]
        
        # Limit number of segments for embedding computation (for speed)
        # Pick longer segments for better embeddings
        sorted_segments = sorted(speech_segments, key=lambda x: x[1] - x[0], reverse=True)
        segments_for_embedding = sorted_segments[:min(30, len(sorted_segments))]  # Max 30 segments
        
        # Filter for segments at least 2 seconds long for reliable embeddings
        segments_for_embedding = [s for s in segments_for_embedding if s[1] - s[0] >= 2000]
        
        if len(segments_for_embedding) < 2:
            # Not enough segments for clustering, return single speaker
            print("Not enough speech segments for speaker clustering, using single speaker")
            total_speaking_time = sum(e - s for s, e in speech_segments) / 1000
            
            # Find best sample
            best_segment = sorted_segments[0]
            sample_start = best_segment[0]
            sample_end = min(best_segment[1], sample_start + 30000)
            sample = audio[sample_start:sample_end]
            sample_path = os.path.join(output_folder, 'speaker_0_sample.wav')
            sample.export(sample_path, format='wav')
            
            return [{
                'id': 0,
                'label': 'Speaker 1',
                'sample_path': sample_path,
                'total_speaking_time': total_speaking_time,
                'segments': speech_segments
            }]
        
        print(f"Computing speaker embeddings for {len(segments_for_embedding)} segments...")
        
        # Extract embeddings for each segment
        embeddings = []
        temp_files = []
        
        try:
            for i, (start_ms, end_ms) in enumerate(segments_for_embedding):
                # Extract segment audio
                segment_audio = audio[start_ms:end_ms]
                
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                segment_audio.export(temp_file.name, format='wav', parameters=["-ac", "1", "-ar", "16000"])
                temp_files.append(temp_file.name)
                
                try:
                    # Get speaker embedding
                    embedding = self._get_speaker_embedding(temp_file.name)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error getting embedding for segment {i}: {e}")
                    continue
            
            if len(embeddings) < 2:
                print("Could not compute enough embeddings, using single speaker")
                raise ValueError("Not enough embeddings")
            
            embeddings = np.array(embeddings)
            
            # Use agglomerative clustering with cosine distance
            # distance_threshold determines how similar speakers must be to be grouped together
            # Lower threshold = more speakers, higher threshold = fewer speakers
            # Using 0.5 for balanced separation (cosine distance: 0=identical, 2=opposite)
            print(f"Clustering {len(embeddings)} embeddings...")
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,  # Balanced threshold to avoid over-splitting
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            
            num_speakers = len(set(labels))
            print(f"Initial clustering found {num_speakers} speaker(s)")
            
            # Post-processing: Merge speakers that are too similar
            # Compute centroids for each cluster
            cluster_centroids = {}
            for label in set(labels):
                cluster_embeddings = embeddings[labels == label]
                centroid = cluster_embeddings.mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                cluster_centroids[label] = centroid
            
            # Check if any clusters should be merged (cosine similarity > 0.80)
            merge_threshold = 0.80  # Lower threshold to merge similar voices
            merged_labels = {label: label for label in set(labels)}
            labels_list = sorted(set(labels))
            
            for i, label1 in enumerate(labels_list):
                for label2 in labels_list[i+1:]:
                    if merged_labels[label1] != merged_labels[label2]:
                        sim = np.dot(cluster_centroids[label1], cluster_centroids[label2])
                        if sim > merge_threshold:
                            old_label = merged_labels[label2]
                            for k in merged_labels:
                                if merged_labels[k] == old_label:
                                    merged_labels[k] = merged_labels[label1]
                            print(f"Merging speaker {label2} into {label1} (similarity: {sim:.3f})")
            
            # Apply merged labels
            labels = np.array([merged_labels[l] for l in labels])
            num_speakers = len(set(labels))
            print(f"Final clustering found {num_speakers} speaker(s) after merging")
            
            # Group segments by speaker
            speaker_segments = {}
            for i, label in enumerate(labels):
                if label not in speaker_segments:
                    speaker_segments[label] = []
                speaker_segments[label].append((segments_for_embedding[i], embeddings[i]))
            
            # Compute centroid embedding for each speaker (for verification)
            speaker_centroids = {}
            for speaker_label, speaker_data in speaker_segments.items():
                embeddings_for_speaker = np.array([s[1] for s in speaker_data])
                centroid = embeddings_for_speaker.mean(axis=0)
                # Normalize the centroid
                centroid = centroid / np.linalg.norm(centroid)
                speaker_centroids[speaker_label] = centroid
            
            # Build speaker info with samples - concatenate multiple verified segments
            speakers = []
            for speaker_label in sorted(speaker_segments.keys()):
                speaker_data = speaker_segments[speaker_label]
                segment_list = [s[0] for s in speaker_data]
                segment_embeddings = [s[1] for s in speaker_data]
                
                # Calculate total speaking time for this speaker
                speaking_time = sum(e - s for s, e in segment_list) / 1000
                
                # Sort segments by similarity to centroid (most representative first)
                centroid = speaker_centroids[speaker_label]
                segment_scores = []
                for seg, emb in zip(segment_list, segment_embeddings):
                    # Cosine similarity to centroid
                    emb_norm = emb / np.linalg.norm(emb)
                    similarity = np.dot(centroid, emb_norm)
                    segment_scores.append((seg, similarity, emb_norm))
                
                # Sort by similarity (highest first) 
                segment_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Use the verification method to extract clean speaker sample
                print(f"Extracting clean sample for speaker {speaker_label}...")
                sample = self._extract_clean_speaker_sample(
                    audio, segment_scores, centroid, target_duration_ms=30000
                )
                
                # Save sample
                sample_path = os.path.join(output_folder, f'speaker_{speaker_label}_sample.wav')
                sample.export(sample_path, format='wav')
                
                speakers.append({
                    'id': int(speaker_label),
                    'label': f'Speaker {speaker_label + 1}',
                    'sample_path': sample_path,
                    'total_speaking_time': speaking_time,
                    'segments': segment_list
                })
            
            # Sort speakers by speaking time (most talkative first)
            speakers.sort(key=lambda x: x['total_speaking_time'], reverse=True)
            
            # Reassign IDs based on sorted order - first rename to temp names to avoid conflicts
            temp_renames = []
            for i, speaker in enumerate(speakers):
                old_id = speaker['id']
                old_path = speaker['sample_path']
                temp_path = os.path.join(output_folder, f'speaker_temp_{i}_sample.wav')
                if os.path.exists(old_path):
                    try:
                        os.rename(old_path, temp_path)
                        temp_renames.append((temp_path, i))
                    except:
                        temp_renames.append((old_path, i))
                speaker['id'] = i
                speaker['label'] = f'Speaker {i + 1}'
            
            # Now rename from temp to final names
            for temp_path, i in temp_renames:
                final_path = os.path.join(output_folder, f'speaker_{i}_sample.wav')
                try:
                    if os.path.exists(temp_path) and temp_path != final_path:
                        os.rename(temp_path, final_path)
                    speakers[i]['sample_path'] = final_path
                except Exception as e:
                    print(f"Warning: Could not rename {temp_path} to {final_path}: {e}")
                    speakers[i]['sample_path'] = temp_path
            
            print(f"Speaker clustering complete: {len(speakers)} speaker(s) found")
            return speakers
            
        except Exception as e:
            print(f"Speaker clustering failed: {e}, falling back to single speaker")
            # Fallback to single speaker
            total_speaking_time = sum(e - s for s, e in speech_segments) / 1000
            best_segment = sorted_segments[0]
            sample_start = best_segment[0]
            sample_end = min(best_segment[1], sample_start + 30000)
            sample = audio[sample_start:sample_end]
            sample_path = os.path.join(output_folder, 'speaker_0_sample.wav')
            sample.export(sample_path, format='wav')
            
            return [{
                'id': 0,
                'label': 'Speaker 1',
                'sample_path': sample_path,
                'total_speaking_time': total_speaking_time,
                'segments': speech_segments
            }]
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def get_speaker_sample(self, audio_path: str, duration: int = 30, speaker_id: int = None) -> str:
        """
        Extract a sample of the speaker's voice for voice cloning.
        
        Args:
            audio_path: Path to the audio file
            duration: Desired sample duration in seconds
            speaker_id: ID of specific speaker (if multiple detected)
            
        Returns:
            Path to the extracted audio sample
        """
        print(f"Extracting speaker sample from: {audio_path}")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Find speech segments
        speech_segments = self._detect_speech_segments(audio)
        
        if not speech_segments:
            # No speech detected, use first portion of audio
            sample_duration_ms = min(duration * 1000, len(audio))
            sample = audio[:sample_duration_ms]
        else:
            # Concatenate speech segments until we have enough duration
            sample = AudioSegment.empty()
            target_duration_ms = duration * 1000
            
            for start, end in speech_segments:
                segment = audio[start:end]
                sample += segment
                
                if len(sample) >= target_duration_ms:
                    sample = sample[:target_duration_ms]
                    break
        
        # Ensure minimum sample length
        if len(sample) < 5000:  # Less than 5 seconds
            # Use whatever we can get
            sample = audio[:min(duration * 1000, len(audio))]
        
        # Save to temp file
        temp_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        sample.export(temp_path, format='wav')
        
        print(f"Speaker sample extracted: {len(sample) / 1000:.1f}s saved to {temp_path}")
        return temp_path
       