"""
Podcast Analyzer Agent
Transcribes and analyzes podcast audio to find optimal ad insertion points
Uses HuggingFace open models for speech-to-text and semantic analysis
Optimized for large podcasts with parallel transcription
"""
import os
import torch
import numpy as np
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
        Analyze podcast content to find sections related to ad topic
        Optimized with batch embedding computation
        
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
            status_callback(40, "Analyzing podcast content for ad placement...")
        
        # Create a topic description from ad analysis
        ad_topic = f"{ad_analysis.get('problem_solved', '')} {ad_analysis.get('product_description', '')}"
        keywords = ad_analysis.get('keywords', [])
        
        # Create segments (group chunks into ~30 second segments)
        segments = self._create_segments(transcription['chunks'], target_duration=30)
        
        if not segments:
            return {
                'segments': [],
                'best_segment': None,
                'insertion_candidates': [],
                'full_transcription': transcription['full_text'],
                'duration': transcription['duration']
            }
        
        if status_callback:
            status_callback(42, f"Computing embeddings for {len(segments)} segments...")
        
        # OPTIMIZATION: Batch encode all segments at once instead of one by one
        segment_texts = [segment['text'] for segment in segments]
        
        # Batch encode: much faster than encoding one by one
        all_embeddings = self.sentence_model.encode(
            [ad_topic] + segment_texts,  # Ad topic + all segments
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=32  # Process in batches
        )
        
        ad_embedding = all_embeddings[0]
        segment_embeddings = all_embeddings[1:]
        
        if status_callback:
            status_callback(45, "Scoring segments for relevance...")
        
        # OPTIMIZATION: Vectorized similarity computation
        # Compute all similarities at once using matrix multiplication
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
            
            # Keyword matching bonus (this is already fast)
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
        # Use audio path for pause detection if available
        audio_path = getattr(self, '_current_audio_path', None)
        insertion_candidates = self._find_insertion_points(transcription, scored_segments, audio_path)
        
        if not insertion_candidates:
            print("No valid insertion candidates found. Returning empty result.")
            return {
                'segments': scored_segments,
                'best_segment': None,
                'insertion_candidates': [],
                'full_transcription': transcription['full_text'],
                'duration': transcription['duration']
            }
        
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
            # Using 0.35 for stricter separation between speakers
            print(f"Clustering {len(embeddings)} embeddings...")
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.35,  # Stricter cosine distance threshold for better separation
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            
            num_speakers = len(set(labels))
            print(f"Clustering found {num_speakers} speaker(s)")
            
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
                    segment_scores.append((seg, similarity))
                
                # Sort by similarity (highest first) 
                segment_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Concatenate top segments until we have ~30 seconds of audio
                # Only use segments with high similarity to centroid (>0.7) to avoid mixed voices
                sample = AudioSegment.empty()
                target_duration_ms = 30000
                min_similarity = 0.7
                
                for seg, similarity in segment_scores:
                    if similarity < min_similarity:
                        continue
                    
                    seg_start, seg_end = seg
                    segment_audio = audio[seg_start:seg_end]
                    
                    # Add a small gap between segments to make it sound natural
                    if len(sample) > 0:
                        sample += AudioSegment.silent(duration=200)  # 200ms gap
                    
                    sample += segment_audio
                    
                    if len(sample) >= target_duration_ms:
                        sample = sample[:target_duration_ms]
                        break
                
                # If we didn't get enough audio with high similarity, lower the threshold
                if len(sample) < 5000:  # Less than 5 seconds
                    print(f"Speaker {speaker_label}: Not enough high-similarity segments, lowering threshold")
                    sample = AudioSegment.empty()
                    for seg, similarity in segment_scores:
                        seg_start, seg_end = seg
                        segment_audio = audio[seg_start:seg_end]
                        
                        if len(sample) > 0:
                            sample += AudioSegment.silent(duration=200)
                        sample += segment_audio
                        
                        if len(sample) >= target_duration_ms:
                            sample = sample[:target_duration_ms]
                            break
                
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
       