"""
Orchestrator for Podcast Ad Integration Pipeline
Coordinates all agents to process podcast and integrate ads
Optimized with parallel processing for speed
"""
import os
import tempfile
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents import (
    AdScriptAnalyzer,
    PodcastAnalyzer,
    VoiceCloningAgent,
    AudioIntegrator,
    TransitionGenerator
)

class PodcastAdOrchestrator:
    def __init__(
        self,
        podcast_path: str,
        ad_script_path: str,
        output_folder: str,
        job_id: str,
        status_callback: Optional[Callable[[int, str], None]] = None,
        speed_mode: str = 'balanced',
        selected_speaker: int = None,
        audio_path: str = None,
        workflow: str = 'full',  # 'full' for integration, 'ad_only' for just ad generation
        speaker_sample_path: str = None  # Pre-saved speaker sample from speaker detection
    ):
        """
        Initialize the orchestrator
        
        Args:
            podcast_path: Path to the podcast audio/video file
            ad_script_path: Path to the ad script text file
            output_folder: Folder to save output files
            job_id: Unique identifier for this job
            status_callback: Callback function(progress, message) for status updates
            speed_mode: 'fast' (quick but less accurate), 'balanced' (default), 'accurate' (slower but better)
            selected_speaker: ID of speaker to use for voice cloning (from speaker detection)
            audio_path: Pre-extracted audio path (if already extracted during speaker detection)
            workflow: 'full' for full integration, 'ad_only' for just generating the ad
            speaker_sample_path: Pre-saved speaker sample path from speaker detection phase
        """
        self.podcast_path = podcast_path
        self.ad_script_path = ad_script_path
        self.output_folder = output_folder
        self.job_id = job_id
        self.status_callback = status_callback or (lambda p, m: None)
        self.speed_mode = speed_mode
        self.selected_speaker = selected_speaker
        self.pre_extracted_audio = audio_path
        self.workflow = workflow
        self.speaker_sample_path = speaker_sample_path  # Pre-saved speaker sample
        
        # Initialize agents with speed mode
        self.ad_analyzer = AdScriptAnalyzer()
        self.podcast_analyzer = PodcastAnalyzer(speed_mode=speed_mode)
        self.voice_cloner = VoiceCloningAgent()
        self.audio_integrator = AudioIntegrator()
        
        # Determine if input is video
        self.is_video = self._is_video_file(podcast_path)
        
        # Create job output folder
        self.job_output_folder = os.path.join(output_folder, job_id)
        os.makedirs(self.job_output_folder, exist_ok=True)
        
        # Store results for retrieval
        self.voice_similarity_score = None
        self.ad_placement_info = None  # Will store ad position details
    
    def _is_video_file(self, path: str) -> bool:
        """Check if file is a video"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        return os.path.splitext(path)[1].lower() in video_extensions
    
    def _update_status(self, progress: int, message: str):
        """Update job status"""
        print(f"[{progress}%] {message}")
        self.status_callback(progress, message)
    
    def process(self) -> str:
        """
        Run the full pipeline to integrate ad into podcast
        Optimized with parallel processing where possible
        
        Returns:
            Path to the output file
        """
        self._update_status(5, "Starting optimized pipeline...")
        
        # ============= PHASE 1: Parallel Initialization =============
        # Ad analysis and audio extraction can run in parallel
        # Use pre-extracted audio if available (from speaker detection)
        audio_path = self.pre_extracted_audio or self.podcast_path
        ad_analysis = None
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            # Start ad analysis
            self._update_status(8, "Analyzing ad script...")
            futures['ad'] = executor.submit(self.ad_analyzer.analyze, self.ad_script_path)
            
            # Start audio extraction if video and not already extracted
            if self.is_video and not self.pre_extracted_audio:
                self._update_status(10, "Extracting audio from video...")
                futures['audio'] = executor.submit(self.podcast_analyzer.extract_audio, self.podcast_path)
            
            # Wait for results
            ad_analysis = futures['ad'].result()
            if 'audio' in futures:
                audio_path = futures['audio'].result()
        
        print(f"Ad Analysis Results:")
        print(f"  Product: {ad_analysis['product_name']}")
        print(f"  Problem: {ad_analysis['problem_solved']}")
        print(f"  Keywords: {ad_analysis['keywords'][:5]}")
        
        # ============= PHASE 2: Transcription (main bottleneck) =============
        self._update_status(15, "Transcribing podcast (this may take a while for long podcasts)...")
        transcription = self.podcast_analyzer.transcribe(
            audio_path,
            status_callback=self.status_callback
        )
        
        # ============= PHASE 3: Parallel Analysis & Sample Extraction =============
        # Content analysis and speaker sample extraction can run in parallel
        content_analysis = None
        speaker_sample = None
        
        # Check if we have a pre-saved speaker sample from speaker detection
        if self.speaker_sample_path and os.path.exists(self.speaker_sample_path):
            print(f"Using pre-saved speaker sample: {self.speaker_sample_path}")
            speaker_sample = self.speaker_sample_path
            # Only need to run content analysis
            self._update_status(40, "Analyzing content (using pre-selected speaker sample)...")
            content_analysis = self.podcast_analyzer.analyze_content(
                transcription,
                ad_analysis,
                self.status_callback,
                audio_path
            )
        else:
            # Run both content analysis and sample extraction in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                self._update_status(40, "Analyzing content and extracting speaker sample...")
                
                futures = {
                    'content': executor.submit(
                        self.podcast_analyzer.analyze_content,
                        transcription,
                        ad_analysis,
                        self.status_callback,
                        audio_path  # Pass audio path for pause detection
                    ),
                    # Pass selected speaker ID for sample extraction
                    'sample': executor.submit(
                        self.podcast_analyzer.get_speaker_sample,
                        audio_path,
                        30,  # duration
                        self.selected_speaker  # speaker_id (None if not selected)
                    )
                }
                
                content_analysis = futures['content'].result()
                speaker_sample = futures['sample'].result()
        
        # Step 4: Determine best insertion point
        insertion_point = self._select_insertion_point(content_analysis)
        print(f"Selected insertion point: {insertion_point['timestamp']:.2f}s")
        print(f"Relevance score: {insertion_point['relevance_score']:.3f}")
        
        # Step 5: Generate ad script with transitions
        self._update_status(50, "Preparing ad script with transitions...")
        full_ad_script = self._create_ad_with_transitions(
            ad_analysis,
            content_analysis,
            insertion_point
        )
        
        # Step 7: Generate ad audio in speaker's voice
        self._update_status(60, "Generating ad audio in podcaster's voice...")
        ad_audio_path = os.path.join(self.job_output_folder, 'generated_ad.wav')
        ad_audio_path = self.voice_cloner.generate_ad_audio(
            full_ad_script,
            speaker_sample,
            ad_audio_path,
            status_callback=self.status_callback
        )
        
        # Get voice similarity score
        self.voice_similarity_score = self.voice_cloner.get_similarity_score()
        
        # Store ad audio path for ad-only workflow
        self.ad_audio_path = ad_audio_path
        
        # If ad-only workflow, return the ad audio path directly
        if self.workflow == 'ad_only':
            self._update_status(100, "Ad generation complete!")
            # Clean up temp files (only if not pre-saved sample)
            if speaker_sample != self.speaker_sample_path:
                try:
                    os.unlink(speaker_sample)
                except:
                    pass
            return ad_audio_path
        
        # Step 8: Integrate ad into podcast (full workflow)
        self._update_status(80, "Integrating ad into podcast...")
        
        # Get ad duration for placement info
        from pydub import AudioSegment
        ad_audio = AudioSegment.from_file(ad_audio_path)
        ad_duration_seconds = len(ad_audio) / 1000.0
        
        # Get original podcast duration
        original_audio = AudioSegment.from_file(audio_path)
        original_duration_seconds = len(original_audio) / 1000.0
        
        # Determine output path
        ext = os.path.splitext(self.podcast_path)[1]
        if self.is_video:
            # For video, we'll first create integrated audio, then merge with video
            integrated_audio_path = os.path.join(
                self.job_output_folder, 
                'integrated_audio.wav'
            )
            
            integrated_audio_path = self.audio_integrator.integrate_ad(
                audio_path,
                ad_audio_path,
                insertion_point['timestamp'],
                output_path=integrated_audio_path
            )
            
            self._update_status(90, "Merging audio with video...")
            output_path = os.path.join(
                self.job_output_folder,
                f'integrated_podcast{ext}'
            )
            
            output_path = self.audio_integrator.integrate_ad_into_video(
                self.podcast_path,
                integrated_audio_path,
                output_path,
                status_callback=self.status_callback
            )
        else:
            # For audio-only, directly integrate
            output_path = os.path.join(
                self.job_output_folder,
                f'integrated_podcast{ext}'
            )
            
            output_path = self.audio_integrator.integrate_ad(
                self.podcast_path,
                ad_audio_path,
                insertion_point['timestamp'],
                output_path=output_path
            )
        
        # Store ad placement info for UI visualization
        self.ad_placement_info = {
            'insertion_point': insertion_point['timestamp'],
            'ad_duration': ad_duration_seconds,
            'ad_end': insertion_point['timestamp'] + ad_duration_seconds,
            'original_duration': original_duration_seconds,
            'final_duration': original_duration_seconds + ad_duration_seconds,
            'relevance_score': insertion_point.get('relevance_score', 0),
            'is_natural_break': insertion_point.get('is_natural_break', False)
        }
        
        # Clean up temp files
        if self.is_video and audio_path != self.podcast_path:
            try:
                os.unlink(audio_path)
            except:
                pass
        
        # Clean up speaker sample only if not pre-saved
        if speaker_sample != self.speaker_sample_path:
            try:
                os.unlink(speaker_sample)
            except:
                pass
        
        self._update_status(100, "Processing complete!")
        
        return output_path
    
    def _select_insertion_point(self, content_analysis: dict) -> dict:
        """
        Select the best insertion point for the ad.
        Prioritizes sentence boundaries to ensure smooth, natural insertion.
        """
        candidates = content_analysis.get('insertion_candidates', [])
        
        if not candidates:
            # Fallback: Insert at 1/3 of the podcast duration
            duration = content_analysis.get('duration', 300)
            return {
                'timestamp': duration / 3,
                'relevance_score': 0.5,
                'is_natural_break': False,
                'context': ''
            }
        
        # Prefer natural breaks (sentence boundaries) with high relevance
        # Candidates are already sorted by (is_natural_break, relevance_score)
        for candidate in candidates:
            if candidate.get('is_natural_break', False):
                print(f"Selected sentence boundary at {candidate['timestamp']:.2f}s")
                if candidate.get('boundary_context'):
                    print(f"  Sentence ends with: ...{candidate['boundary_context'][-50:]}")
                return candidate
        
        # If no natural breaks found, use highest relevance but warn
        print("Warning: No sentence boundary found, using segment end (may cause mid-sentence cut)")
        return candidates[0]
    
    def _create_ad_with_transitions(
        self,
        ad_analysis: dict,
        content_analysis: dict,
        insertion_point: dict
    ) -> str:
        """Create the full ad script with intro and outro transitions"""
        
        # Generate contextual intro
        topic = ad_analysis.get('problem_solved', '')
        intro = TransitionGenerator.generate_intro(topic)
        
        # Get the ad script
        ad_script = ad_analysis.get('full_script', '')
        
        # Generate outro
        outro = TransitionGenerator.generate_outro()
        
        # Combine
        full_script = f"{intro}\n\n{ad_script}\n\n{outro}"
        
        return full_script


# Test function
if __name__ == "__main__":
    print("Orchestrator module loaded successfully!")
    print("To use, create a PodcastAdOrchestrator instance and call process()")
